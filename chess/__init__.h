/* Generated by Cython 0.29.5 */

#ifndef __PYX_HAVE__chess____init__
#define __PYX_HAVE__chess____init__


#ifndef __PYX_HAVE_API__chess____init__

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C unsigned PY_LONG_LONG BB_ALL;

#endif /* !__PYX_HAVE_API__chess____init__ */

/* WARNING: the interface of the module init function changed in CPython 3.5. */
/* It now returns a PyModuleDef instance instead of a PyModule instance. */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initchess(void);
#else
PyMODINIT_FUNC PyInit_chess(void);
#endif

#endif /* !__PYX_HAVE__chess____init__ */
