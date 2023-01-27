/* Generated by Cython 0.29.21 */

#ifndef __PYX_HAVE__faircorels___corels
#define __PYX_HAVE__faircorels___corels

#include "Python.h"

#ifndef __PYX_HAVE_API__faircorels___corels

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

__PYX_EXTERN_C double compute_sample_robustness(int, int, int, int, double, int, int, int);
__PYX_EXTERN_C void print_memo_info_sample_robustness_auditor(void);

#endif /* !__PYX_HAVE_API__faircorels___corels */

/* WARNING: the interface of the module init function changed in CPython 3.5. */
/* It now returns a PyModuleDef instance instead of a PyModule instance. */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_corels(void);
#else
PyMODINIT_FUNC PyInit__corels(void);
#endif

#endif /* !__PYX_HAVE__faircorels___corels */