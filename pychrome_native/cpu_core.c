#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static inline float clampf(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static void kelvin_to_rgb(float kelvin, float out_rgb[3]) {
    if (kelvin < 1000.0f) kelvin = 1000.0f;
    if (kelvin > 40000.0f) kelvin = 40000.0f;

    float tmp = kelvin / 100.0f;
    float red, green, blue;

    if (tmp <= 66.0f) {
        red = 255.0f;
    } else {
        red = 329.698727446f * powf((tmp - 60.0f), -0.1332047592f);
    }

    if (tmp <= 66.0f) {
        green = 99.4708025861f * logf(tmp) - 161.1195681661f;
    } else {
        green = 288.1221695283f * powf((tmp - 60.0f), -0.0755148492f);
    }

    if (tmp >= 66.0f) {
        blue = 255.0f;
    } else if (tmp <= 19.0f) {
        blue = 0.0f;
    } else {
        blue = 138.5177312231f * logf(tmp - 10.0f) - 305.0447927307f;
    }

    out_rgb[0] = clampf(red / 255.0f);
    out_rgb[1] = clampf(green / 255.0f);
    out_rgb[2] = clampf(blue / 255.0f);
}

static PyObject* process_frame_float32(PyObject* self, PyObject* args) {
    PyObject* input_obj = NULL;

    double wb_temp;
    double wb_tint;
    double fracRx;
    double fracGx;
    double fracBY;
    double gammaRx;
    double gammaRy;
    double gammaGx;
    double gammaGy;
    double gammaBY;
    double exposure;

    if (!PyArg_ParseTuple(
            args,
            "Oddddddddddd",
            &input_obj,
            &wb_temp,
            &wb_tint,
            &fracRx,
            &fracGx,
            &fracBY,
            &gammaRx,
            &gammaRy,
            &gammaGx,
            &gammaGy,
            &gammaBY,
            &exposure)) {
        return NULL;
    }

    PyArrayObject* input_arr = (PyArrayObject*)PyArray_FROM_OTF(
        input_obj,
        NPY_FLOAT32,
        NPY_ARRAY_CARRAY_RO
    );
    if (input_arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(input_arr) != 3 || PyArray_DIM(input_arr, 2) < 3) {
        Py_DECREF(input_arr);
        PyErr_SetString(PyExc_ValueError, "rgb must have shape [H, W, 3+]");
        return NULL;
    }

    const npy_intp h = PyArray_DIM(input_arr, 0);
    const npy_intp w = PyArray_DIM(input_arr, 1);
    const npy_intp c = PyArray_DIM(input_arr, 2);

    npy_intp out_dims[3] = {h, w, 3};
    PyArrayObject* out_arr = (PyArrayObject*)PyArray_SimpleNew(3, out_dims, NPY_FLOAT32);
    if (out_arr == NULL) {
        Py_DECREF(input_arr);
        return NULL;
    }

    float src_rgb[3];
    float ref_rgb[3];
    kelvin_to_rgb((float)wb_temp, src_rgb);
    kelvin_to_rgb(6500.0f, ref_rgb);

    const float eps_gain = 1.0e-8f;
    float gain_r = ref_rgb[0] / (src_rgb[0] + eps_gain);
    float gain_g = ref_rgb[1] / (src_rgb[1] + eps_gain);
    float gain_b = ref_rgb[2] / (src_rgb[2] + eps_gain);

    float tint_norm = (float)wb_tint;
    if (tint_norm < -100.0f) tint_norm = -100.0f;
    if (tint_norm > 100.0f) tint_norm = 100.0f;
    tint_norm /= 100.0f;
    gain_g *= (1.0f - 0.15f * tint_norm);

    const float eps = 1.0e-6f;
    float fRx = (float)fracRx;
    float fGx = (float)fracGx;
    float fBy = (float)fracBY;
    if (fRx < eps) fRx = eps;
    if (fGx < eps) fGx = eps;
    if (fBy < eps) fBy = eps;

    const float fRy = 1.0f - fRx;
    const float fGy = 1.0f - fGx;

    float gRx = (float)gammaRx;
    float gRy = (float)gammaRy;
    float gGx = (float)gammaGx;
    float gGy = (float)gammaGy;
    float gBy = (float)gammaBY;
    if (gRx < eps) gRx = eps;
    if (gRy < eps) gRy = eps;
    if (gGx < eps) gGx = eps;
    if (gGy < eps) gGy = eps;
    if (gBy < eps) gBy = eps;

    const float exp_f = (float)exposure;

    const float* in_ptr = (const float*)PyArray_DATA(input_arr);
    float* out_ptr = (float*)PyArray_DATA(out_arr);
    const npy_intp pixels = h * w;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < pixels; ++i) {
        const npy_intp in_idx = i * c;
        const npy_intp out_idx = i * 3;

        float Z1 = clampf(in_ptr[in_idx + 0] * gain_r);
        float Z2 = clampf(in_ptr[in_idx + 1] * gain_g);
        float Z3 = clampf(in_ptr[in_idx + 2] * gain_b);

        float innerY = 1.0f - (Z3 / fBy);
        if (innerY < 0.0f) innerY = 0.0f;
        if (innerY > 1.0f) innerY = 1.0f;
        float Y = 1.0f - powf(innerY, 1.0f / gBy);

        float tmp1 = powf(1.0f - Y, gRy);
        float termR = fRy * (1.0f - tmp1);
        float innerX1 = 1.0f - ((Z1 - termR) / fRx);
        if (innerX1 < 0.0f) innerX1 = 0.0f;
        if (innerX1 > 1.0f) innerX1 = 1.0f;
        float X1 = 1.0f - powf(innerX1, 1.0f / gRx);

        float tmp2 = powf(1.0f - Y, gGy);
        float termG = fGy * (1.0f - tmp2);
        float innerX2 = 1.0f - ((Z2 - termG) / fGx);
        if (innerX2 < 0.0f) innerX2 = 0.0f;
        if (innerX2 > 1.0f) innerX2 = 1.0f;
        float X2 = 1.0f - powf(innerX2, 1.0f / gGx);

        out_ptr[out_idx + 0] = clampf(Y * exp_f);
        out_ptr[out_idx + 1] = clampf(X1 * exp_f);
        out_ptr[out_idx + 2] = clampf(X2 * exp_f);
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(input_arr);
    return (PyObject*)out_arr;
}

static PyMethodDef CpuCoreMethods[] = {
    {
        "process_frame_float32",
        process_frame_float32,
        METH_VARARGS,
        "Process RGB float32 frame with WB + IRG transform in one fused pass."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpu_core_module = {
    PyModuleDef_HEAD_INIT,
    "_cpu_core",
    "Py-Chrome native CPU core",
    -1,
    CpuCoreMethods
};

PyMODINIT_FUNC PyInit__cpu_core(void) {
    import_array();
    return PyModule_Create(&cpu_core_module);
}
