#include "llama-tp.h"

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>

static int64_t tp_time_us() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000; }
#include <unistd.h>
#include <sched.h>

// floats per rank area: covers n_embd (<= 4096) * ubatch (<= 32768) with headroom
#define TP_CAP (128u * 1024u * 1024u)

struct llama_tp_context::shm_hdr {
    uint32_t             magic;
    int32_t              size;
    std::atomic<int32_t> arrive_cnt;
    std::atomic<int32_t> arrive_sense;
    std::atomic<int32_t> release_cnt;
    std::atomic<int32_t> release_sense;
};

static constexpr uint32_t TP_MAGIC = 0x54504c4c; // "LLPT"

static long tp_futex_wait(std::atomic<int32_t> * addr, int32_t expected, int64_t timeout_us = 0) {
    struct timespec ts;
    struct timespec * pts = nullptr;
    if (timeout_us > 0) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        ts.tv_sec  += timeout_us / 1000000;
        ts.tv_nsec += (timeout_us % 1000000) * 1000;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec  += 1;
            ts.tv_nsec -= 1000000000L;
        }
        pts = &ts;
    }
    return syscall(SYS_futex, (void *) addr, FUTEX_WAIT, expected, pts, nullptr, 0);
}

static long tp_futex_wake(std::atomic<int32_t> * addr, int n) {
    return syscall(SYS_futex, (void *) addr, FUTEX_WAKE, n, nullptr, nullptr, 0);
}

int32_t llama_tp_context::expert_lo(int32_t n_expert) const {
    return (n_expert / size) * rank;
}

int32_t llama_tp_context::expert_hi(int32_t n_expert) const {
    return rank == size - 1 ? n_expert : (n_expert / size) * (rank + 1);
}

bool llama_tp_context::init(int32_t size_, int32_t rank_, const char * name) {
    size = size_;
    rank = rank_;
    cap  = TP_CAP;
    if (size <= 1) {
        return true;
    }
    assert(rank >= 0 && rank < size);

    const std::string seg = "/llama_tp_" + std::string(name && name[0] ? name : "default");

    shm_fd = shm_open(seg.c_str(), O_CREAT | O_RDWR, 0600);
    if (shm_fd < 0) {
        fprintf(stderr, "llama_tp: shm_open(%s): %s\n", seg.c_str(), strerror(errno));
        return false;
    }
    map_len = sizeof(shm_hdr) + (size_t) size * cap * sizeof(float);
    if (ftruncate(shm_fd, (off_t) map_len) != 0) {
        fprintf(stderr, "llama_tp: ftruncate: %s\n", strerror(errno));
        return false;
    }
    map_base = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (map_base == MAP_FAILED) {
        fprintf(stderr, "llama_tp: mmap: %s\n", strerror(errno));
        map_base = nullptr;
        return false;
    }
    hdr   = (shm_hdr *) map_base;
    areas = (float *) ((char *) map_base + sizeof(shm_hdr));

    if (rank == 0) {
        // first rank initializes the shared header
        hdr->magic         = TP_MAGIC;
        hdr->size          = size;
        hdr->arrive_cnt    = 0;
        hdr->arrive_sense  = 0;
        hdr->release_cnt   = 0;
        hdr->release_sense = 0;
        __sync_synchronize();
    }
    // wait for rank 0's header init (both ranks may open the segment around the same time)
    for (int i = 0; i < 10000; i++) {
        if (hdr->magic == TP_MAGIC && hdr->size == size) {
            break;
        }
        usleep(1000);
    }
    if (hdr->magic != TP_MAGIC || hdr->size != size) {
        fprintf(stderr, "llama_tp: shm segment mismatch (stale segment? unlink %s and retry)\n", seg.c_str());
        return false;
    }
    return true;
}

void llama_tp_context::finalize() {
    if (map_base != nullptr) {
        munmap(map_base, map_len);
        map_base = nullptr;
    }
    if (shm_fd >= 0) {
        close(shm_fd);
        shm_fd = -1;
    }
}

void llama_tp_context::allreduce(float * data, size_t n) {
    if (size <= 1) {
        return;
    }
    assert(n <= cap);

    float * my = areas + (size_t) rank * cap;
    memcpy(my, data, n * sizeof(float));

    const int32_t s = sense ? 1 : 0;

    // phase 1 (arrive): every rank's data is in the segment once all ranks have arrived
    static int64_t round = 0;
    const int64_t rnd = ++round;
    if (hdr->arrive_cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == size) {
        hdr->arrive_sense.store(s, std::memory_order_release);
        hdr->arrive_cnt.store(0, std::memory_order_release); // reset for the next round
        tp_futex_wake(&hdr->arrive_sense, size);
    } else {
        const int64_t t0 = tp_time_us();
        while (hdr->arrive_sense.load(std::memory_order_acquire) != s) {
            // spin-yield: the partner rank is computing on other cores and arrives within
            //   microseconds; no futex (its timeout is unreliable here and the lost-wakeup
            //   race it leaves is the deadlock we hit)
            sched_yield();
            if (tp_time_us() - t0 > 10 * 1000 * 1000) {
                fprintf(stderr, "llama_tp: rank %d round %lld STALLED at arrive (want %d, sense %d, cnt %d)\n",
                        rank, (long long) rnd, s, hdr->arrive_sense.load(std::memory_order_acquire),
                        hdr->arrive_cnt.load(std::memory_order_acquire));
                abort();
            }
        }
    }

    // combine: every rank sums the areas in rank order -> bit-identical results across ranks
    for (int32_t r = 0; r < size; r++) {
        if (r == rank) {
            continue;
        }
        const float * other = areas + (size_t) r * cap;
        for (size_t i = 0; i < n; i++) {
            data[i] += other[i];
        }
    }

    // phase 2 (release): no rank may overwrite its area until every rank finished summing
    if (hdr->release_cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == size) {
        hdr->release_sense.store(s, std::memory_order_release);
        hdr->release_cnt.store(0, std::memory_order_release);
        tp_futex_wake(&hdr->release_sense, size);
    } else {
        const int64_t t0 = tp_time_us();
        while (hdr->release_sense.load(std::memory_order_acquire) != s) {
            sched_yield();
            if (tp_time_us() - t0 > 10 * 1000 * 1000) {
                fprintf(stderr, "llama_tp: rank %d round %lld STALLED at release (want %d, sense %d, cnt %d)\n",
                        rank, (long long) rnd, s, hdr->release_sense.load(std::memory_order_acquire),
                        hdr->release_cnt.load(std::memory_order_acquire));
                abort();
            }
        }
    }

    sense = !sense;
}
