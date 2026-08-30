package proxy

import "syscall"

// syscallECONNREFUSED is named so isHardFailure reads without a platform
// import at its call site.
const syscallECONNREFUSED = syscall.ECONNREFUSED
