"""Private entry point. Preload arrays, close input files, then forbid file/network access."""
import resource
import sys


def main():
    cpu, memory = map(int, sys.argv[1:])
    # Importing from network storage can take long; the calculation's own limit is set after the imports.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu + 120, cpu + 120))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import base64
    import ctypes
    import errno
    import json
    import socket  # Preload the module; creating sockets is still forbidden below.
    import numpy as np
    from scipy import ndimage as ndi

    # These are the only files the trusted setup reads. All are private scratch inputs.
    with open('spec.json') as file:
        spec = json.load(file)
    with open('code.txt') as file:
        code = file.read()
    arrays = {}
    with np.load('input.npz', allow_pickle=False) as data:
        for key in data.files:
            a = data[key]
            # Immutable bytes backing prevents setflags(write=True) from changing an input.
            arrays[key] = np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
    output = open('output.json', 'w')
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7fff0000)  # allow existing array operations
    if not context:
        raise RuntimeError('Cannot initialize Python syscall sandbox')
    deny = 0x00050000 | errno.EPERM
    # No new files, sockets, processes or access to another process's memory.
    names = ('open', 'openat', 'openat2', 'creat', 'socket', 'socketpair', 'connect', 'bind',
             'execve', 'execveat', 'fork', 'vfork', 'clone', 'clone3', 'ptrace',
             'process_vm_readv', 'process_vm_writev', 'pidfd_getfd', 'kill', 'tkill', 'tgkill',
             'io_uring_setup', 'open_by_handle_at', 'mount', 'unshare', 'setns',
             'unlink', 'unlinkat', 'rename', 'renameat', 'renameat2', 'mkdir', 'mkdirat', 'rmdir',
             'link', 'linkat', 'symlink', 'symlinkat', 'truncate', 'chmod', 'fchmodat',
             'chown', 'lchown', 'fchownat', 'setxattr', 'lsetxattr', 'removexattr', 'lremovexattr')
    try:
        for name in names:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(context, deny, number, 0) != 0:
                raise RuntimeError('Cannot install Python syscall rule')
        if lib.seccomp_load(context) != 0:
            raise RuntimeError('Cannot activate Python syscall sandbox')
    finally:
        lib.seccomp_release(context)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    limit = int(usage.ru_utime + usage.ru_stime) + 1 + cpu
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    # Drop setup paths/arguments before any generated code executes.
    sys.argv = ['array_worker']
    saved = []
    def save_mask(array):
        if len(saved) >= 32:
            raise ValueError('At most 32 output masks per call')
        array = np.asarray(array)
        if array.shape != tuple(spec['shape']):
            raise ValueError('save_mask needs an image-sized array')
        saved.append(base64.b64encode(np.packbits(array.astype(bool).ravel())).decode('ascii'))
        return len(saved) - 1
    namespace = dict(np=np, ndi=ndi, masks={k: arrays[k] for k in spec['masks']},
        instances={k: arrays[k] for k in spec['instances']}, images={k: arrays[k] for k in ('t1', 't2')},
        save_mask=save_mask)
    try:
        exec(compile(code, '<array_tool>', 'exec'), namespace)
        result = dict(masks=saved)
    except BaseException as exc:
        result = dict(error=f'{type(exc).__name__}: {exc}')
    json.dump(result, output)
    output.close()


if __name__ == '__main__':
    main()
