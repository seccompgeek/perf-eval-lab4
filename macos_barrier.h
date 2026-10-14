#ifndef LAB4_MACOS_BARRIER_H
#define LAB4_MACOS_BARRIER_H

/* macOS has pthread mutexes and conditions, but no POSIX barriers. */
#ifdef __APPLE__
#include <errno.h>
#include <pthread.h>

#define PTHREAD_BARRIER_SERIAL_THREAD (-1)
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned count, waiting, generation;
} pthread_barrier_t;

static int pthread_barrier_init(pthread_barrier_t *barrier,
                                const void *attr, unsigned count) {
    if (attr || count == 0) return EINVAL;
    int rc = pthread_mutex_init(&barrier->mutex, NULL);
    if (rc) return rc;
    rc = pthread_cond_init(&barrier->condition, NULL);
    if (rc) {
        pthread_mutex_destroy(&barrier->mutex);
        return rc;
    }
    barrier->count = count;
    barrier->waiting = barrier->generation = 0;
    return 0;
}

static int pthread_barrier_wait(pthread_barrier_t *barrier) {
    int rc = pthread_mutex_lock(&barrier->mutex);
    if (rc) return rc;
    unsigned generation = barrier->generation;
    if (++barrier->waiting == barrier->count) {
        barrier->waiting = 0;
        ++barrier->generation;
        rc = pthread_cond_broadcast(&barrier->condition);
        pthread_mutex_unlock(&barrier->mutex);
        return rc ? rc : PTHREAD_BARRIER_SERIAL_THREAD;
    }
    while (generation == barrier->generation) {
        rc = pthread_cond_wait(&barrier->condition, &barrier->mutex);
        if (rc) break;
    }
    pthread_mutex_unlock(&barrier->mutex);
    return rc;
}

static int pthread_barrier_destroy(pthread_barrier_t *barrier) {
    int rc = pthread_cond_destroy(&barrier->condition);
    if (rc) return rc;
    return pthread_mutex_destroy(&barrier->mutex);
}
#endif
#endif
