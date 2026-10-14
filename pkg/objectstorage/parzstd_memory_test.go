// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package objectstorage

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

func TestChunkFor(t *testing.T) {
	cases := []struct {
		name string
		in   int64
		want int
	}{
		{"empty", 0, parZstdMinChunk},
		{"tiny metadata file", 36, parZstdMinChunk},
		{"just under the floor", parZstdMinChunk - parZstdChunkSlack - 1, parZstdMinChunk},
		{"above the floor keeps slack", 1 << 20, (1 << 20) + parZstdChunkSlack},
		{"just under a full chunk", parZstdChunk - parZstdChunkSlack - 1, parZstdChunk - 1},
		{"exactly a full chunk", parZstdChunk, parZstdChunk},
		{"larger than a full chunk", 64 << 20, parZstdChunk},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := chunkFor(tc.in); got != tc.want {
				t.Errorf("chunkFor(%d) = %d, want %d", tc.in, got, tc.want)
			}
			if got := chunkFor(tc.in); got > parZstdChunk {
				t.Errorf("chunkFor(%d) = %d exceeds parZstdChunk %d", tc.in, got, parZstdChunk)
			}
		})
	}
}

// A small input must not reserve a full-size frame buffer. Before #2553 the pool
// was filled eagerly with workers*parZstdQueue buffers of parZstdChunk each, so a
// 36-byte file reserved 16 MiB.
func TestSmallInputDoesNotReserveFullChunk(t *testing.T) {
	p := newParZstd(io.Discard, 1, chunkFor(36))
	defer p.Close()
	if got := cap(p.buf); got != parZstdMinChunk {
		t.Errorf("first buffer cap = %d, want %d", got, parZstdMinChunk)
	}
	// Only the one buffer in hand has been created; the rest of the pool's
	// allowance is still unspent.
	if got, want := len(p.quota), 1*parZstdQueue-1; got != want {
		t.Errorf("unspent buffer allowance = %d, want %d", got, want)
	}
}

// The pool still refuses to grow past its cap, so the lazy fill did not turn the
// bound into an unbounded allocator.
func TestBufferPoolStillCapped(t *testing.T) {
	p := newParZstd(io.Discard, 2, parZstdMinChunk)
	defer p.Close()
	// One buffer is already held in p.buf; drain the rest of the allowance.
	created := 1
	for len(p.quota) > 0 {
		<-p.quota
		created++
	}
	if want := 2 * parZstdQueue; created != want {
		t.Errorf("pool allowance = %d buffers, want %d", created, want)
	}
}

// Round-trip at the sizes where the new chunk sizing changes behavior.
func TestSparseZstdRoundTripAtChunkBoundaries(t *testing.T) {
	sizes := []int64{
		0,
		36,
		parZstdMinChunk - 1,
		parZstdMinChunk,
		parZstdChunk - parZstdChunkSlack - 1,
		parZstdChunk,
		parZstdChunk + 1,
		2*parZstdChunk + 12345,
	}
	for _, size := range sizes {
		t.Run(name(size), func(t *testing.T) {
			dir := t.TempDir()
			srcPath := filepath.Join(dir, "src")
			var regions []region
			if size > 0 {
				regions = []region{{0, size, 0x5A}}
			}
			writeSparseSource(t, srcPath, size, regions)
			src, err := os.Open(srcPath)
			if err != nil {
				t.Fatal(err)
			}
			defer src.Close()

			var buf bytes.Buffer
			logical, _, err := writeSparseZstd(&buf, src)
			if err != nil {
				t.Fatalf("writeSparseZstd: %v", err)
			}
			if logical != size {
				t.Errorf("logical = %d, want %d", logical, size)
			}

			dstPath := filepath.Join(dir, "dst")
			dst, err := os.Create(dstPath)
			if err != nil {
				t.Fatal(err)
			}
			defer dst.Close()
			got, err := readSparseZstd(dst, bytes.NewReader(buf.Bytes()[len(sparseMagic):]))
			if err != nil {
				t.Fatalf("readSparseZstd: %v", err)
			}
			if got != size {
				t.Errorf("readSparseZstd size = %d, want %d", got, size)
			}
			want, err := os.ReadFile(srcPath)
			if err != nil {
				t.Fatal(err)
			}
			have, err := os.ReadFile(dstPath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(have, want) {
				t.Fatalf("round-trip mismatch at size %d (got %d bytes, want %d)", size, len(have), len(want))
			}
		})
	}
}

// Concurrent compression of tiny files must scale with the input, not with
// parZstdChunk. The reporter measured 397.8 MiB for 24 files of 36 bytes.
func TestConcurrentSmallFileAllocation(t *testing.T) {
	const files = 24
	dir := t.TempDir()
	paths := make([]string, files)
	for i := range paths {
		p := filepath.Join(dir, "f"+name(int64(i)))
		if err := os.WriteFile(p, make([]byte, 36), 0o600); err != nil {
			t.Fatal(err)
		}
		paths[i] = p
	}
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	var wg sync.WaitGroup
	for _, p := range paths {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			f, err := os.Open(p)
			if err != nil {
				t.Error(err)
				return
			}
			defer f.Close()
			if _, err := writeContent(io.Discard, f); err != nil {
				t.Error(err)
			}
		}(p)
	}
	wg.Wait()
	runtime.ReadMemStats(&after)

	got := after.TotalAlloc - before.TotalAlloc
	// Generously above the ~17 MiB observed after the fix and far below the
	// ~398 MiB before it, so this fails on a regression without being brittle.
	const limit = 96 << 20
	t.Logf("%d concurrent 36-byte files allocated %.1f MiB", files, float64(got)/(1<<20))
	if got > limit {
		t.Errorf("allocated %d bytes for %d tiny files, want under %d", got, files, limit)
	}
}

func name(n int64) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
