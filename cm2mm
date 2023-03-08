#include <unordered_map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <intrin.h>
#include <immintrin.h>
#include <iostream>
#include <any>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <direct.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#define _GNU_SOURCE      
#include <math.h>        

#ifndef CM2_BLOCK_SIZE
#define CM2_BLOCK_SIZE (8)
#endif
#include <filesystem>

#define PYTHONSUPPORT extern "C"
#ifndef CM2MMAPI
#define CM2MMAPI PYTHONSUPPORT
#endif

#define _CM2MM_MVC_NAME_ "__mvc__"
#define _CM2MM_MVC_ERROR_DEF_ "0"

#ifdef _CM2MM_EXCEPTION_LANGUAGE_ENGLISH
#define _CM2MM_MVC_ERROR_INVALID_EXPRESSION "invalid expression"
#define _CM2MM_MVC_ERROR_INVALID_EXPRESSION_UNKNOWN_VAR_FUNC(_VARIABLE) \
"Invalid expression: unknown variable or function " + _VARIABLE
#define _CM2MM_MVC_ERROR_USUPORTED_OPERATION "Unsupported opration"
#define _CM2MM_MVC_ERROR_VECTOR_HAVE_INCORRECT_SIZE \
"The transposition vector has an incompatible size with the size of the matrix - \
the size should be equal to the height of the matrix"
#define _CM2MM_MVC_ERROR_INCORRECT_MATRIX_SIZE "Matrix sizes do not match"
#else
#define _CM2MM_MVC_ERROR_INVALID_EXPRESSION "Неверно задано выражение"
#define _CM2MM_MVC_ERROR_INVALID_EXPRESSION_UNKNOWN_VAR_FUNC(_VARIABLE) \
"Неверно задано выражение: неизвестное название переменной или функции " + _VARIABLE
#define _CM2MM_MVC_ERROR_USUPORTED_OPERATION "Данная операция не поддерживается"
#define _CM2MM_MVC_ERROR_VECTOR_HAVE_INCORRECT_SIZE \
"Вектор транспозиции имеет несовместимый размер с размером матрицы - \
размер должен быть равен высоте матрицы"
#define _CM2MM_MVC_ERROR_INCORRECT_MATRIX_SIZE "Размеры матриц не совпадают"
#endif

typedef enum _cm2_simd_support : unsigned char
{
	CM2_SIMD_SUPPORT_EMPTY = 0x00,
	CM2_SIMD_SUPPORT_AVX,
	CM2_SIMD_SUPPORT_AVX2,
} _cm2_simd_support;

struct _cm2
{
	_cm2();

	_cm2(unsigned int width, unsigned int height);

	_cm2(unsigned int width, unsigned int height, const float* data);

	unsigned int width;

	unsigned int height;

	unsigned int block_width;

	unsigned int block_height;

	unsigned int byte_size;

	unsigned int byte_width;

	unsigned int byte_height;

	unsigned int size;

	unsigned int physical_size;

	unsigned int blocks_count;

	float* data;

	float* data_ptr_end;

	//~_cm2();
};

struct _cv2
{
	_cv2();

	_cv2(unsigned int width);

	_cv2(unsigned int width, const float* data);

	//~_cv2();

	unsigned int width;


	unsigned int block_width;


	unsigned int byte_size;

	unsigned int byte_width;

	unsigned int size;

	unsigned int physical_size;

	unsigned int blocks_count;

	float* data;

	float* data_ptr_end;
};



_cm2 cm2_create(unsigned int width, unsigned int height);

void cm2_set11(_cm2* matrix, float value);

void cm2_set1(_cm2* matrix, const float* value);

void cm2_set(_cm2* matrix, const float* data);

void cm2_load(_cm2* matrix, const float* data);

void cm2_store(float* data, _cm2* matrix);

void cm2_add(_cm2* dst, const _cm2* src_a, const _cm2* src_b);

void cm2_add(_cm2* dst, const _cm2* src_a, const float src_b);

void cm2_sin(_cm2* dst, const _cm2* src);

void cm2_cos(_cm2* dst, const _cm2* src);

void cm2_tan(_cm2* dst, const _cm2* src);

void cm2_sub(_cm2* dst, const _cm2* src_a, const _cm2* src_b);

void cm2_sub(_cm2* dst, const _cm2* src_a, const float src_b);

// scale matrix
void cm2_scl(_cm2* dst, const _cm2* src_a, const float src_b);

// scale matrix by vector of transposition
void cm2_stp(_cm2* dst, const _cm2* src_a, const _cv2* src_b);

void cm2_tsp(_cm2* dst, const _cm2* src);

void cm2_inv(_cm2* dst, const _cm2* src);

// replase strings
void cm2_rps(_cm2* dst, const _cm2* src, int idx_f, int idx_s);

void cm2_mul(_cm2* dst, const _cm2* src_a, const _cm2* src_b);

//void cm2_mul_ps_avx2_mt(_cm2* dst, const _cm2* src_a, const _cm2* src_b);


void cm2_set_simd_support(_cm2_simd_support);

unsigned _cm2_get_block_size(unsigned size);


void cv2_load(_cv2* dst, const float* data);

void cv2_set1(_cv2* dst, const float value);

void cv2_set(_cv2* dst, const float* data);

void cv2_add(_cv2* dst, const _cv2* src_a, const _cv2* src_b);

void cv2_add(_cv2* dst, const _cv2* src_a, const float src_b);

void cv2_sin(_cv2* dst, const _cv2* src);

void cv2_cos(_cv2* dst, const _cv2* src);

void cv2_tan(_cv2* dst, const _cv2* src);

void cv2_sub(_cv2* dst, const _cv2* src_a, const _cv2* src_b);

void cv2_sub(_cv2* dst, const _cv2* src_a, const float src_b);

void cv2_scl(_cv2* dst, const _cv2* src_a, const float src_b);

void cv2_rps(_cv2* dst, const _cv2* src, int idx_f, int idx_s);

void cv2_inv(_cv2* dst, const _cv2* src);

_cm2_simd_support cm2_get_simd_support();


static _cm2_simd_support cm2_simd_support = CM2_SIMD_SUPPORT_EMPTY;

float _cs2_fmadd(const float src_a, const float src_b, const float src_c)
{
	float result = src_a * src_b + src_c;
	return result;
}

void _cs2_store(float* dst, const float src)
{
	*dst = src;
}

float* _cm2_ddata_begin(_cm2* matrix)
{
	return matrix->data;
}

__m128* _cm2_ddata_begin_avx(_cm2* matrix)
{
	return (__m128*)matrix->data;
}

__m256* _cm2_ddata_begin_avx2(_cm2* matrix)
{
	return (__m256*)matrix->data;
}

float* _cm2_ddata_end(_cm2* matrix)
{
	return matrix->data_ptr_end;
}

__m128* _cm2_ddata_end_avx(_cm2* matrix)
{
	return (__m128*)matrix->data_ptr_end;
}

__m256* _cm2_ddata_end_avx2(_cm2* matrix)
{
	return (__m256*)matrix->data_ptr_end;
}

const float* _cm2_sdata_begin(const _cm2* matrix)
{
	return matrix->data;
}

const __m128* _cm2_sdata_begin_avx(const _cm2* matrix)
{
	return (__m128*)matrix->data;
}

const __m256* _cm2_sdata_begin_avx2(const _cm2* matrix)
{
	return (__m256*)matrix->data;
}

const float* _cm2_sdata_end(const _cm2* matrix)
{
	return matrix->data_ptr_end;
}

const __m128* _cm2_sdata_end_avx(const _cm2* matrix)
{
	return (__m128*)matrix->data_ptr_end;
}

const __m256* _cm2_sdata_end_avx2(const _cm2* matrix)
{
	return (__m256*)matrix->data_ptr_end;
}


float* _cv2_ddata_begin(_cv2* vector)
{
	return vector->data;
}

__m128* _cv2_ddata_begin_avx(_cv2* vector)
{
	return (__m128*)vector->data;
}

__m256* _cv2_ddata_begin_avx2(_cv2* vector)
{
	return (__m256*)vector->data;
}

float* _cv2_ddata_end(_cv2* vector)
{
	return vector->data_ptr_end;
}

__m128* _cv2_ddata_end_avx(_cv2* vector)
{
	return (__m128*)vector->data_ptr_end;
}

__m256* _cv2_ddata_end_avx2(_cv2* vector)
{
	return (__m256*)vector->data_ptr_end;
}

const float* _cv2_sdata_begin(const _cv2* vector)
{
	return vector->data;
}

const __m128* _cv2_sdata_begin_avx(const _cv2* vector)
{
	return (__m128*)vector->data;
}

const __m256* _cv2_sdata_begin_avx2(const _cv2* vector)
{
	return (__m256*)vector->data;
}

const float* _cv2_sdata_end(const _cv2* vector)
{
	return vector->data_ptr_end;
}

const __m128* _cv2_sdata_end_avx(const _cv2* vector)
{
	return (__m128*)vector->data_ptr_end;
}

const __m256* _cv2_sdata_end_avx2(const _cv2* vector)
{
	return (__m256*)vector->data_ptr_end;
}


// first row end templated
template <typename _Value_type>
constexpr inline _Value_type* _cm2_ddata_frend_t(_cm2* matrix)
{
	return (_Value_type*)(matrix->data + matrix->block_width);
}

// dst first row end ref
float* _cm2_ddata_frend(_cm2* matrix)
{
	return _cm2_ddata_frend_t<float>(matrix);
}

// dst first row end avx
__m128* _cm2_ddata_frend_avx(_cm2* matrix)
{
	return _cm2_ddata_frend_t<__m128>(matrix);
}

// dst first row end avx2
__m256* _cm2_ddata_frend_avx2(_cm2* matrix)
{
	return _cm2_ddata_frend_t<__m256>(matrix);
}

// dst first row end ref
const float* _cm2_sdata_frend(_cm2* matrix)
{
	return _cm2_ddata_frend_t<const float>(matrix);
}

// dst first row end avx
const __m128* _cm2_sdata_frend_avx(_cm2* matrix)
{
	return _cm2_ddata_frend_t<const __m128>(matrix);
}

// dst first row end avx2
const __m256* _cm2_sdata_frend_avx2(_cm2* matrix)
{
	return _cm2_ddata_frend_t<const __m256>(matrix);
}

float* _cm2_ddata_next_row(float* data_ptr, _cm2* matrix)
{
	return data_ptr + matrix->width;//
}

float* _cm2_ddata_get_row(float* data_ptr, _cm2* matrix, unsigned row_index)
{
	return data_ptr + matrix->block_width * row_index;
}

const float* _cm2_sdata_get_row(const float* data_ptr, const _cm2* matrix, unsigned row_index)
{
	return data_ptr + matrix->width * row_index;//
}

const float* _cm2_sdata_next_row(const float* data_ptr, const _cm2* matrix)
{
	return data_ptr + matrix->block_width;
}

__m128* _cm2_ddata_next_row_avx(__m128* data_ptr, _cm2* matrix)
{
	return reinterpret_cast<__m128*>((reinterpret_cast<float*>(data_ptr) + matrix->block_width));
}

const __m128* _cm2_sdata_next_row_avx(const __m128* data_ptr, const _cm2* matrix)
{
	return reinterpret_cast<const __m128*>((reinterpret_cast<const float*>(data_ptr) + matrix->block_width));
}

__m256* _cm2_ddata_next_row_avx2(__m256* data_ptr, _cm2* matrix)
{
	return reinterpret_cast<__m256*>((reinterpret_cast<float*>(data_ptr) + matrix->block_width));
}

const __m256* _cm2_sdata_next_row_avx2(const __m256* data_ptr, const _cm2* matrix)
{
	return reinterpret_cast<const __m256*>((reinterpret_cast<const float*>(data_ptr) + matrix->block_width));
}

template<int _Offset = 1, typename _Data_type>
_Data_type* _cm2_ddata_next_col(_Data_type* data_ptr)
{
	return data_ptr + _Offset;
}

template<int _Offset = 1, typename _Data_type>
const _Data_type* _cm2_sdata_next_col(const _Data_type* data_ptr)
{
	return data_ptr + _Offset;
}

auto _cm2_get_byte_size(_cm2* matrix)
{
	auto size = static_cast<unsigned long long>(matrix->block_width) *
		matrix->block_height * sizeof(float);
	return size;
}

auto _cv2_get_byte_size(_cv2* vector)
{
	auto size = static_cast<unsigned long long>(vector->block_width) * sizeof(float);
	return size;
}

unsigned _cm2_get_block_size(unsigned size, unsigned block)
{
	auto block_size = (size + (block - 1)) & (-static_cast<int>(block));
	return block_size;
}

unsigned _cm2_get_block_size(unsigned size)
{
	auto block_size = _cm2_get_block_size(size, CM2_BLOCK_SIZE);
	return block_size;
}

_cm2 cm2_create(unsigned int width, unsigned int height)
{
	return _cm2(width, height);
}

void _cm2_set11_empty(_cm2* matrix, float value)
{
	float* data_ptr = _cm2_ddata_begin(matrix);
	float* data_ptr_end = _cm2_ddata_end(matrix);
	while (data_ptr <= data_ptr_end)
	{
		_cs2_store(data_ptr, value);
		data_ptr = _cm2_ddata_next_col(data_ptr);
	}
}

void _cm2_set11_avx(_cm2* matrix, float value)
{
	float* data_ptr = _cm2_ddata_begin(matrix);
	float* data_ptr_end = _cm2_ddata_end(matrix);
	__m128 _value_row_1x4 = _mm_set1_ps(value);
	while (data_ptr < data_ptr_end)
	{
		_mm_storeu_ps(data_ptr, _value_row_1x4);
		data_ptr = _cm2_ddata_next_col<4>(data_ptr);
	}
}

void _cm2_set11_avx2(_cm2* matrix, float value)
{
	float* data_ptr = _cm2_ddata_begin(matrix);
	float* data_ptr_end = _cm2_ddata_end(matrix);
	__m256 _value_row_1x8 = _mm256_set1_ps(value);
	while (data_ptr < data_ptr_end)
	{
		_mm256_storeu_ps(data_ptr, _value_row_1x8);
		data_ptr = _cm2_ddata_next_col<8>(data_ptr);
	}
}

void cm2_set11(_cm2* matrix, float value)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_set11_empty(matrix, value);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_set11_avx(matrix, value);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_set11_avx2(matrix, value);
		break;
	default:
		break;
	}
}

void _cm2_set1_memcpy(_cm2* matrix, const float* values)
{
	float* data_ptr = _cm2_ddata_begin(matrix);
	float* data_ptr_end = _cm2_ddata_end(matrix);
	while (data_ptr < data_ptr_end)
	{
		memcpy(data_ptr, values, matrix->byte_width);
		_cm2_ddata_next_row(data_ptr, matrix);
	}
}

void _cm2_set1_empty(_cm2* matrix, const float* values)
{
	_cm2_set1_memcpy(matrix, values);
}

void _cm2_set1_avx(_cm2* matrix, const float* values)
{
	_cm2_set1_memcpy(matrix, values);
}

void _cm2_set1_avx2(_cm2* matrix, const float* values)
{
	_cm2_set1_memcpy(matrix, values);
}

void cm2_set1(_cm2* matrix, const float* values)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_set1_empty(matrix, values);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_set1_avx(matrix, values);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_set1_avx2(matrix, values);
		break;
	default:
		_cm2_set1_memcpy(matrix, values);
		break;
	}
}

void cm2_set(_cm2* matrix, const float* data)
{
	cm2_load(matrix, data);
}

void _cm2_load_memcpy(_cm2* matrix, const float* data)
{
	float* data_ptr = _cm2_ddata_begin(matrix);
	float* data_ptr_end = _cm2_ddata_end(matrix);
	while (data_ptr < data_ptr_end)
	{
		memcpy(data_ptr, data, matrix->byte_width);
		data += matrix->width;
		data_ptr = _cm2_ddata_next_row(data_ptr, matrix);
	}
}

void _cm2_load_empty(_cm2* matrix, const float* data)
{
	_cm2_load_memcpy(matrix, data);
}

void _cm2_load_avx(_cm2* matrix, const float* data)
{
	_cm2_load_memcpy(matrix, data);
}

void _cm2_load_avx2(_cm2* matrix, const float* data)
{
	_cm2_load_memcpy(matrix, data);
}

void cm2_load(_cm2* matrix, const float* data)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_load_empty(matrix, data);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_load_avx(matrix, data);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_load_avx2(matrix, data);
		break;
	default:
		_cm2_load_memcpy(matrix, data);
		break;
	}
}

void _cv2_load_memcpy(_cv2* vector, const float* data)
{
	float* data_ptr = _cv2_ddata_begin(vector);
	memcpy(data_ptr, data, vector->byte_width);
}

void _cv2_load_empty(_cv2* vector, const float* data)
{
	_cv2_load_memcpy(vector, data);
}

void _cv2_load_avx(_cv2* vector, const float* data)
{
	_cv2_load_memcpy(vector, data);
}

void _cv2_load_avx2(_cv2* vector, const float* data)
{
	_cv2_load_memcpy(vector, data);
}

void cv2_load(_cv2* vector, const float* data)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_load_empty(vector, data);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_load_avx(vector, data);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_load_avx2(vector, data);
		break;
	default:
		_cv2_load_memcpy(vector, data);
		break;
	}
}

void _cv2_set1_empty(_cv2* vector, const float value)
{
	float* data_ptr = _cv2_ddata_begin(vector);
	float* data_ptr_end = _cv2_ddata_end(vector);
	while (data_ptr < data_ptr_end)
	{
		*data_ptr = value;
		data_ptr++;
	}
}

void _cv2_set1_avx(_cv2* vector, const float value)
{
	__m128* data_ptr = _cv2_ddata_begin_avx(vector);
	__m128* data_ptr_end = _cv2_ddata_end_avx(vector);
	while (data_ptr < data_ptr_end)
	{
		*data_ptr = _mm_set_ps1(value);
		data_ptr++;
	}
}

void _cv2_set1_avx2(_cv2* vector, const float value)
{
	__m256* data_ptr = _cv2_ddata_begin_avx2(vector);
	__m256* data_ptr_end = _cv2_ddata_end_avx2(vector);
	while (data_ptr < data_ptr_end)
	{
		*data_ptr = _mm256_set1_ps(value);
		data_ptr++;
	}
}

void cv2_set1(_cv2* vector, const float value)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_set1_empty(vector, value);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_set1_avx(vector, value);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_set1_avx2(vector, value);
		break;
	default:
		_cv2_set1_empty(vector, value);
		break;
	}
}

void cv2_set(_cv2* vector, const float* data)
{
	cv2_load(vector, data);
}

inline void _cv2_add_empty(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	const float* src_b_ptr = _cv2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr + *src_b_ptr;
		dst_ptr++;
		src_a_ptr++;
		src_b_ptr++;
	}
}

inline void _cv2_add_avx(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	const float* src_b_ptr = _cv2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_src_b_row_1x4 = _mm_loadu_ps(src_b_ptr);
		_sum_row_1x4 = _mm_add_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_b_ptr += 4;
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

inline void _cv2_add_avx2(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	const float* src_b_ptr = _cv2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_src_b_row_1x8 = _mm256_loadu_ps(src_b_ptr);
		_sum_row_1x8 = _mm256_add_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_b_ptr += 8;
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cv2_add(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_add_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_add_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_add_avx2(dst, src_a, src_b);
		break;
	default:
		_cv2_add_empty(dst, src_a, src_b);
		break;
	}
}

inline void _cv2_add_empty(_cv2* dst, const _cv2* src_a, const float src_b)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr + src_b;
		dst_ptr++;
		src_a_ptr++;
	}
}

inline void _cv2_add_avx(_cv2* dst, const _cv2* src_a, const float src_b)
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	_src_b_row_1x4 = _mm_set_ps1(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_sum_row_1x4 = _mm_add_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

inline void _cv2_add_avx2(_cv2* dst, const _cv2* src_a, const float src_b)
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	_src_b_row_1x8 = _mm256_set1_ps(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_sum_row_1x8 = _mm256_add_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cv2_add(_cv2* dst, const _cv2* src_a, const float src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_add_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_add_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_add_avx2(dst, src_a, src_b);
		break;
	default:
		_cv2_add_empty(dst, src_a, src_b);
		break;
	}
}

inline void _cv2_sin_empty(_cv2* dst, const _cv2* src)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_ptr = _cv2_sdata_begin(src);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = sin(*src_ptr);
		dst_ptr++;
		src_ptr++;
	}
}

inline void _cv2_sin_avx(_cv2* dst, const _cv2* src)
{
	_cv2_sin_empty(dst, src);
}

inline void _cv2_sin_avx2(_cv2* dst, const _cv2* src)
{
	_cv2_sin_empty(dst, src);
}

inline void cv2_sin(_cv2* dst, const _cv2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_sin_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_sin_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_sin_avx2(dst, src);
		break;
	default:
		_cv2_sin_empty(dst, src);
		break;
	}
}

inline void _cv2_cos_empty(_cv2* dst, const _cv2* src)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_ptr = _cv2_sdata_begin(src);
#pragma omp simd
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = cos(*src_ptr);
		dst_ptr++;
		src_ptr++;
	}
}

inline void _cv2_cos_avx(_cv2* dst, const _cv2* src)
{
	_cv2_cos_empty(dst, src);
}

inline void _cv2_cos_avx2(_cv2* dst, const _cv2* src)
{
	_cv2_cos_empty(dst, src);
}

inline void cv2_cos(_cv2* dst, const _cv2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_cos_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_cos_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_cos_avx2(dst, src);
		break;
	default:
		_cv2_cos_empty(dst, src);
		break;
	}
}

inline void _cv2_tan_empty(_cv2* dst, const _cv2* src)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_ptr = _cv2_sdata_begin(src);
#pragma omp simd
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = tan(*src_ptr);
		dst_ptr++;
		src_ptr++;
	}
}

inline void _cv2_tan_avx(_cv2* dst, const _cv2* src)
{
	_cv2_tan_empty(dst, src);
}

inline void _cv2_tan_avx2(_cv2* dst, const _cv2* src)
{
	_cv2_tan_empty(dst, src);
}

inline void cv2_tan(_cv2* dst, const _cv2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_tan_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_tan_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_tan_avx2(dst, src);
		break;
	default:
		_cv2_tan_empty(dst, src);
		break;
	}
}

inline void _cv2_sub_empty(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	const float* src_b_ptr = _cv2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{

		*dst_ptr = *src_a_ptr - *src_b_ptr;
		dst_ptr++;
		src_a_ptr++;
		src_b_ptr++;
	}
}

inline void _cv2_sub_avx(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	const float* src_b_ptr = _cv2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_src_b_row_1x4 = _mm_loadu_ps(src_b_ptr);
		_sum_row_1x4 = _mm_sub_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_b_ptr += 4;
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

inline void _cv2_sub_avx2(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	const float* src_b_ptr = _cv2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_src_b_row_1x8 = _mm256_loadu_ps(src_b_ptr);
		_sum_row_1x8 = _mm256_sub_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_b_ptr += 8;
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cv2_sub(_cv2* dst, const _cv2* src_a, const _cv2* src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_sub_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_sub_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_sub_avx2(dst, src_a, src_b);
		break;
	default:
		_cv2_sub_empty(dst, src_a, src_b);
		break;
	}
}

inline void _cv2_sub_empty(_cv2* dst, const _cv2* src_a, const float src_b)
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr - src_b;
		dst_ptr++;
		src_a_ptr++;
	}
}

inline void _cv2_sub_avx(_cv2* dst, const _cv2* src_a, const float src_b)
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	_src_b_row_1x4 = _mm_set_ps1(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_sum_row_1x4 = _mm_sub_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

inline void _cv2_sub_avx2(_cv2* dst, const _cv2* src_a, const float src_b)
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	_src_b_row_1x8 = _mm256_set1_ps(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_sum_row_1x8 = _mm256_sub_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cv2_sub(_cv2* dst, const _cv2* src_a, const float src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_sub_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_sub_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_sub_avx2(dst, src_a, src_b);
		break;
	default:
		_cv2_sub_empty(dst, src_a, src_b);
		break;
	}
}

void _cv2_scl_empty(_cv2* dst, const _cv2* src_a, const float src_b) noexcept
{
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr * src_b;
		src_a_ptr++;
		dst_ptr++;
	}
}

void _cv2_scl_avx(_cv2* dst, const _cv2* src_a, const float src_b) noexcept
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	_src_b_row_1x4 = _mm_set_ps1(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_sum_row_1x4 = _mm_mul_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

void _cv2_scl_avx2(_cv2* dst, const _cv2* src_a, const float src_b) noexcept
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cv2_ddata_begin(dst);
	float* dst_ptr_end = _cv2_ddata_end(dst);
	const float* src_a_ptr = _cv2_sdata_begin(src_a);
	_src_b_row_1x8 = _mm256_set1_ps(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_sum_row_1x8 = _mm256_mul_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cv2_scl(_cv2* dst, const _cv2* src_a, const float src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_scl_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_scl_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_scl_avx2(dst, src_a, src_b);
		break;
	default:
		_cv2_scl_empty(dst, src_a, src_b);
		break;
	}
}

inline void _cv2_rsp_empty(_cv2* dst, const _cv2* src, int idx_f, int idx_s)
{
	for (int x = 0; x < src->width; x++)
	{
		if (x == idx_f)
		{
			dst->data[idx_s] = src->data[idx_f];
		}
		else if (x == idx_s)
		{
			dst->data[idx_f] = src->data[idx_s];
		}
		else
		{
			dst->data[x] = src->data[x];
		}
	}
}

inline void cv2_rps(_cv2* dst, const _cv2* src, int idx_f, int idx_s)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_rsp_empty(dst, src, idx_f, idx_s);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_rsp_empty(dst, src, idx_f, idx_s);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_rsp_empty(dst, src, idx_f, idx_s);
		break;
	default:
		_cv2_rsp_empty(dst, src, idx_f, idx_s);
		break;
	}
}

inline void _cv2_inv_empty(_cv2* dst, const _cv2* src)
{
	unsigned offset_width = src->width - 1;
	for (int x = 0; x < src->width; x++)
	{
		dst->data[x] = src->data[offset_width - x];
	}
}

inline void cv2_inv(_cv2* dst, const _cv2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cv2_inv_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cv2_inv_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cv2_inv_empty(dst, src);
		break;
	default:
		_cv2_inv_empty(dst, src);
		break;
	}
}

void _cm2_store_memcpy(float* dst, _cm2* matrix)
{
	const float* data_ptr = _cm2_ddata_begin(matrix);
	const float* data_ptr_end = _cm2_ddata_end(matrix);
	while (data_ptr < data_ptr_end)
	{
		memcpy(dst, data_ptr, matrix->byte_width);
		dst = _cm2_ddata_next_row(dst, matrix);
		data_ptr = _cm2_sdata_next_row(data_ptr, matrix);
	}
}

void _cm2_store_empty(float* dst, _cm2* matrix)
{
	_cm2_store_memcpy(dst, matrix);
}

void _cm2_store_avx(float* dst, _cm2* matrix)
{
	_cm2_store_memcpy(dst, matrix);
}

void _cm2_store_avx2(float* dst, _cm2* matrix)
{
	_cm2_store_memcpy(dst, matrix);
}

void cm2_store(float* dst, _cm2* matrix)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_store_empty(dst, matrix);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_store_avx(dst, matrix);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_store_avx2(dst, matrix);
		break;
	default:
		_cm2_store_memcpy(dst, matrix);
		break;
	}
}

void _cm2_add_empty(_cm2* dst, const _cm2* src_a, const _cm2* src_b) noexcept
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	const float* src_b_ptr = _cm2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr + *src_b_ptr;
		src_b_ptr++;
		src_a_ptr++;
		dst_ptr++;
	}
}

void _cm2_add_avx(_cm2* dst, const _cm2* src_a, const _cm2* src_b) noexcept
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	const float* src_b_ptr = _cm2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_src_b_row_1x4 = _mm_loadu_ps(src_b_ptr);
		_sum_row_1x4 = _mm_add_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_b_ptr += 4;
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

void _cm2_add_avx2(_cm2* dst, const _cm2* src_a, const _cm2* src_b) noexcept
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	const float* src_b_ptr = _cm2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_src_b_row_1x8 = _mm256_loadu_ps(src_b_ptr);
		_sum_row_1x8 = _mm256_add_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_b_ptr += 8;
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

void cm2_add(_cm2* dst, const _cm2* src_a, const _cm2* src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_add_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_add_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_add_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_add_empty(dst, src_a, src_b);
		break;
	}
}

void _cm2_add_empty(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr + src_b;
		src_a_ptr++;
		dst_ptr++;
	}
}

void _cm2_add_avx(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	_src_b_row_1x4 = _mm_set_ps1(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_sum_row_1x4 = _mm_add_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

void _cm2_add_avx2(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	_src_b_row_1x8 = _mm256_set1_ps(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_sum_row_1x8 = _mm256_add_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cm2_add(_cm2* dst, const _cm2* src_a, const float src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_add_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_add_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_add_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_add_empty(dst, src_a, src_b);
		break;
	}
}

inline void _cm2_sin_empty(_cm2* dst, const _cm2* src)
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_ptr = _cm2_sdata_begin(src);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = sin(*src_ptr);
		dst_ptr++;
		src_ptr++;
	}
}

inline void _cm2_sin_avx(_cm2* dst, const _cm2* src)
{
	_cm2_sin_empty(dst, src);
}

inline void _cm2_sin_avx2(_cm2* dst, const _cm2* src)
{
	_cm2_sin_empty(dst, src);
}

inline void cm2_sin(_cm2* dst, const _cm2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_sin_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_sin_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_sin_avx2(dst, src);
		break;
	default:
		_cm2_sin_empty(dst, src);
		break;
	}
}

inline void _cm2_cos_empty(_cm2* dst, const _cm2* src)
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_ptr = _cm2_sdata_begin(src);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = cos(*src_ptr);
		dst_ptr++;
		src_ptr++;
	}
}

inline void _cm2_cos_avx(_cm2* dst, const _cm2* src)
{
	_cm2_cos_empty(dst, src);
}

inline void _cm2_cos_avx2(_cm2* dst, const _cm2* src)
{
	_cm2_cos_empty(dst, src);
}

inline void cm2_cos(_cm2* dst, const _cm2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_cos_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_cos_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_cos_avx2(dst, src);
		break;
	default:
		_cm2_cos_empty(dst, src);
		break;
	}
}

inline void _cm2_tan_empty(_cm2* dst, const _cm2* src)
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_ptr = _cm2_sdata_begin(src);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = tan(*src_ptr);
		dst_ptr++;
		src_ptr++;
	}
}

inline void _cm2_tan_avx(_cm2* dst, const _cm2* src)
{
	_cm2_tan_empty(dst, src);
}

inline void _cm2_tan_avx2(_cm2* dst, const _cm2* src)
{
	_cm2_tan_empty(dst, src);
}

inline void cm2_tan(_cm2* dst, const _cm2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_tan_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_tan_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_tan_avx2(dst, src);
		break;
	default:
		_cm2_tan_empty(dst, src);
		break;
	}
}


void _cm2_sub_empty(_cm2* dst, const _cm2* src_a, const _cm2* src_b) noexcept
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	const float* src_b_ptr = _cm2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr - *src_b_ptr;
		src_b_ptr++;
		src_a_ptr++;
		dst_ptr++;
	}
}

void _cm2_sub_avx(_cm2* dst, const _cm2* src_a, const _cm2* src_b) noexcept
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	const float* src_b_ptr = _cm2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_src_b_row_1x4 = _mm_loadu_ps(src_b_ptr);
		_sum_row_1x4 = _mm_sub_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_b_ptr += 4;
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

void _cm2_sub_avx2(_cm2* dst, const _cm2* src_a, const _cm2* src_b) noexcept
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	const float* src_b_ptr = _cm2_sdata_begin(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_src_b_row_1x8 = _mm256_loadu_ps(src_b_ptr);
		_sum_row_1x8 = _mm256_sub_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_b_ptr += 8;
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

void cm2_sub(_cm2* dst, const _cm2* src_a, const _cm2* src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_sub_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_sub_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_sub_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_sub_empty(dst, src_a, src_b);
		break;
	}
}

void _cm2_sub_empty(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr - src_b;
		src_a_ptr++;
		dst_ptr++;
	}
}

void _cm2_sub_avx(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	_src_b_row_1x4 = _mm_set_ps1(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_sum_row_1x4 = _mm_sub_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

void _cm2_sub_avx2(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	_src_b_row_1x8 = _mm256_set1_ps(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_sum_row_1x8 = _mm256_sub_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

inline void cm2_sub(_cm2* dst, const _cm2* src_a, const float src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_sub_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_sub_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_sub_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_sub_empty(dst, src_a, src_b);
		break;
	}
}

void _cm2_scl_empty(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	while (dst_ptr < dst_ptr_end)
	{
		*dst_ptr = *src_a_ptr * src_b;
		src_a_ptr++;
		dst_ptr++;
	}
}

void _cm2_scl_avx(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	__m128					_src_a_row_1x4, _src_b_row_1x4;
	__m128					_sum_row_1x4;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	_src_b_row_1x4 = _mm_set_ps1(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x4 = _mm_loadu_ps(src_a_ptr);
		_sum_row_1x4 = _mm_mul_ps(_src_a_row_1x4, _src_b_row_1x4);
		_mm_storeu_ps(dst_ptr, _sum_row_1x4);
		src_a_ptr += 4;
		dst_ptr += 4;
	}
}

void _cm2_scl_avx2(_cm2* dst, const _cm2* src_a, const float src_b) noexcept
{
	__m256					_src_a_row_1x8, _src_b_row_1x8;
	__m256					_sum_row_1x8;
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_a_ptr = _cm2_sdata_begin(src_a);
	_src_b_row_1x8 = _mm256_set1_ps(src_b);
	while (dst_ptr < dst_ptr_end)
	{
		_src_a_row_1x8 = _mm256_loadu_ps(src_a_ptr);
		_sum_row_1x8 = _mm256_mul_ps(_src_a_row_1x8, _src_b_row_1x8);
		_mm256_storeu_ps(dst_ptr, _sum_row_1x8);
		src_a_ptr += 8;
		dst_ptr += 8;
	}
}

void cm2_scl(_cm2* dst, const _cm2* src_a, const float src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_scl_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_scl_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_scl_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_scl_empty(dst, src_a, src_b);
		break;
	}
}

void _cm2_tsp_empty(_cm2* dst, const _cm2* src)
{
	for (unsigned n = 0; n < dst->width * dst->height; n++)
	{
		unsigned i = n / dst->height;
		unsigned j = n % dst->height;
		dst->data[n] = src->data[dst->width * j + i];
	}
}

inline void _cm2_tsp4x4_avx_la(
	float* dst_data, const float* src_data,
	const unsigned int block_width, const unsigned int block_height) noexcept
{
	__m128 _row0_1x4, _row1_1x4, _row2_1x4, _row3_1x4;

	_row0_1x4 = _mm_load_ps(&src_data[0 * block_width]);
	_row1_1x4 = _mm_load_ps(&src_data[1 * block_width]);
	_row2_1x4 = _mm_load_ps(&src_data[2 * block_width]);
	_row3_1x4 = _mm_load_ps(&src_data[3 * block_width]);

	__m128 _shfl_r01_44, _shfl_r01_EE, _shfl_r23_44, _shfl_r23_EE;

	_shfl_r01_44 = _mm_shuffle_ps(_row0_1x4, _row1_1x4, 0x44);
	_shfl_r23_44 = _mm_shuffle_ps(_row0_1x4, _row1_1x4, 0xEE);
	_shfl_r01_EE = _mm_shuffle_ps(_row2_1x4, _row3_1x4, 0x44);
	_shfl_r23_EE = _mm_shuffle_ps(_row2_1x4, _row3_1x4, 0xEE);

	_row0_1x4 = _mm_shuffle_ps(_shfl_r01_44, _shfl_r01_EE, 0x88);
	_row1_1x4 = _mm_shuffle_ps(_shfl_r01_44, _shfl_r01_EE, 0xDD);
	_row2_1x4 = _mm_shuffle_ps(_shfl_r23_44, _shfl_r23_EE, 0x88);
	_row3_1x4 = _mm_shuffle_ps(_shfl_r23_44, _shfl_r23_EE, 0xDD);

	_mm_store_ps(&dst_data[0 * block_height], _row0_1x4);
	_mm_store_ps(&dst_data[1 * block_height], _row1_1x4);
	_mm_store_ps(&dst_data[2 * block_height], _row2_1x4);
	_mm_store_ps(&dst_data[3 * block_height], _row3_1x4);
}

inline void _cm2_tsp8x8_avx2_ra(
	float* dst_data,
	const float* src_data,
	const unsigned int block_width,
	const unsigned int block_height) noexcept
{
	__m256  r0, r1, r2, r3, r4, r5, r6, r7;
	__m256  t0, t1, t2, t3, t4, t5, t6, t7;

	r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[0 * block_width + 0])), _mm_load_ps(&src_data[4 * block_width + 0]), 1);
	r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[1 * block_width + 0])), _mm_load_ps(&src_data[5 * block_width + 0]), 1);
	r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[2 * block_width + 0])), _mm_load_ps(&src_data[6 * block_width + 0]), 1);
	r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[3 * block_width + 0])), _mm_load_ps(&src_data[7 * block_width + 0]), 1);
	r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[0 * block_width + 4])), _mm_load_ps(&src_data[4 * block_width + 4]), 1);
	r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[1 * block_width + 4])), _mm_load_ps(&src_data[5 * block_width + 4]), 1);
	r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[2 * block_width + 4])), _mm_load_ps(&src_data[6 * block_width + 4]), 1);
	r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&src_data[3 * block_width + 4])), _mm_load_ps(&src_data[7 * block_width + 4]), 1);

	t0 = _mm256_unpacklo_ps(r0, r1);
	t1 = _mm256_unpackhi_ps(r0, r1);
	t2 = _mm256_unpacklo_ps(r2, r3);
	t3 = _mm256_unpackhi_ps(r2, r3);
	t4 = _mm256_unpacklo_ps(r4, r5);
	t5 = _mm256_unpackhi_ps(r4, r5);
	t6 = _mm256_unpacklo_ps(r6, r7);
	t7 = _mm256_unpackhi_ps(r6, r7);

	__m256 v;

	r0 = _mm256_shuffle_ps(t0, t2, 0x44);
	r1 = _mm256_shuffle_ps(t0, t2, 0xEE);
	r2 = _mm256_shuffle_ps(t1, t3, 0x44);
	r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
	r4 = _mm256_shuffle_ps(t4, t6, 0x44);
	r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
	r6 = _mm256_shuffle_ps(t5, t7, 0x44);
	r7 = _mm256_shuffle_ps(t5, t7, 0xEE);


	_mm256_store_ps(&dst_data[0 * block_height], r0);
	_mm256_store_ps(&dst_data[1 * block_height], r1);
	_mm256_store_ps(&dst_data[2 * block_height], r2);
	_mm256_store_ps(&dst_data[3 * block_height], r3);
	_mm256_store_ps(&dst_data[4 * block_height], r4);
	_mm256_store_ps(&dst_data[5 * block_height], r5);
	_mm256_store_ps(&dst_data[6 * block_height], r6);
	_mm256_store_ps(&dst_data[7 * block_height], r7);
}

void _cm2_tsp_avx(_cm2* dst, const _cm2* src) noexcept
{
	const auto max_w = src->width;
	const auto max_h = src->height;
	for (auto x = 0U; x < max_w; x += 4U)
	{
		for (auto y = 0U; y < max_h; y += 4U)
		{
			_cm2_tsp4x4_avx_la(
				&dst->data[x * max_w + y],
				&src->data[y * max_h + x],
				max_w,
				max_h
			);
		}
	}
}

void _cm2_tsp_avx2(_cm2* dst, const _cm2* src)
{
	const auto max_w = src->width;
	const auto max_h = src->height;

	for (unsigned i = 0U; i < max_w; i += 32U) {
		for (unsigned j = 0U; j < max_h; j += 32U) {
			unsigned max_i2 = i + 32U < max_w ? i + 32U : max_w;
			unsigned max_j2 = j + 32U < max_h ? j + 32U : max_h;
			for (unsigned i2 = i; i2 < max_i2; i2 += 8U) {
				for (unsigned j2 = j; j2 < max_j2; j2 += 8U) {
					_cm2_tsp8x8_avx2_ra(
						&dst->data[i2 * max_w + j2],
						&src->data[j2 * max_h + i2],
						max_w,
						max_h
					);
				}
			}
		}
	}
}

void _cm2_stp_empty(_cm2* dst, const _cm2* src_a, const _cv2* src_b)
{
	const float* vector_ptr = _cv2_sdata_begin(src_b);
	const float* vector_ptr_end = _cv2_sdata_end(src_b);
	float* dst_ptr = _cm2_ddata_begin(dst);
	float* dst_ptr_end = _cm2_ddata_end(dst);
	const float* src_ptr_begin = _cm2_sdata_begin(src_a);
	while (vector_ptr < vector_ptr_end)
	{
		const float* src_row = _cm2_sdata_get_row(src_ptr_begin, src_a, *vector_ptr);
		memcpy(dst_ptr, src_row, 16ULL);
		vector_ptr++;
		dst_ptr = _cm2_ddata_next_row(dst_ptr, dst);
	}
}

void _cm2_stp_avx(_cm2* dst, const _cm2* src_a, const _cv2* src_b)
{
	_cm2_stp_empty(dst, src_a, src_b);
}

void _cm2_stp_avx2(_cm2* dst, const _cm2* src_a, const _cv2* src_b)
{
	_cm2_stp_empty(dst, src_a, src_b);
}

void cm2_stp(_cm2* dst, const _cm2* src_a, const _cv2* src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_stp_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_stp_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_stp_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_stp_empty(dst, src_a, src_b);
		break;
	}
}

void cm2_tsp(_cm2* dst, const _cm2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_tsp_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_tsp_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_tsp_avx2(dst, src);
		break;
	default:
		_cm2_tsp_empty(dst, src);
		break;
	}
}

inline void _cm2_inv_empty(_cm2* dst, const _cm2* src)
{
	// создаем копию матрицы mat
	_cm2 src_copy = _cm2(src->width, src->height, src->data);

	// создаем единичную матрицу того же размера, что и исходная матрица
	_cm2 single = _cm2(src->width, src->height);
	float* single_ptr = _cm2_ddata_begin(&single);
	float* single_ptr_end = _cm2_ddata_end(&single);
	const unsigned single_ptr_offset = single.width + 1;
	while (single_ptr < single_ptr_end)
	{
		*single_ptr = 1;
		single_ptr += single_ptr_offset;
	}

	// приводим исходную матрицу и единичную матрицу к верхнетреугольному виду
	for (int j = 0; j < src->width; j++)
	{
		for (int i = 0; i < src->width; i++)
		{
			if (i != j)
			{
				float ratio = src_copy.data[i * src->width + j] / src_copy.data[j * src->width + j];
				for (int k = 0; k < src->width; k++)
				{
					src_copy.data[i * src->width + k] -= ratio * src_copy.data[j * src->width + k];
					single.data[i * src->width + k] -= ratio * single.data[j * src->width + k];
				}
			}
		}
	}

	// делим каждый элемент строки на соответствующий диагональный элемент
	for (int i = 0; i < src->width; i++)
	{
		float temp = src_copy.data[i * src->width + i];
		for (int j = 0; j < src->width; j++)
		{
			src_copy.data[i * src->width + j] /= temp;
			single.data[i * src->width + j] /= temp;
		}
	}

	//копируем единичную матрицу в матрицу-результат
	for (int i = 0; i < src->width; i++)
	{
		for (int j = 0; j < src->width; j++)
		{
			dst->data[i * src->width + j] = single.data[i * src->width + j];
		}
	}
}

inline void _cm2_inv_avx(_cm2* dst, const _cm2* src)
{
	_cm2_inv_empty(dst, src);
}

inline void _cm2_inv_avx2(_cm2* dst, const _cm2* src)
{
	_cm2_inv_empty(dst, src);
}

inline void cm2_inv(_cm2* dst, const _cm2* src)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_inv_empty(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_inv_avx(dst, src);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_inv_avx2(dst, src);
		break;
	default:
		_cm2_inv_empty(dst, src);
		break;
	}
}

void _cm2_rps_empty(_cm2* dst, const _cm2* src, int idx_f, int idx_s)
{
	for (int y = 0; y < src->height; y++)
	{
		if (y == idx_f)
		{
			memcpy(&dst->data[idx_s * dst->width], &src->data[idx_f * dst->width], dst->byte_width);
		}
		else if (y == idx_s)
		{
			memcpy(&dst->data[idx_f * dst->width], &src->data[idx_s * dst->width], dst->byte_width);
		}
		else
		{
			memcpy(&dst->data[y * dst->width], &src->data[y * dst->width], dst->byte_width);
		}
	}
}

void cm2_rps(_cm2* dst, const _cm2* src, int idx_f, int idx_s)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_rps_empty(dst, src, idx_f, idx_s);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_rps_empty(dst, src, idx_f, idx_s);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_rps_empty(dst, src, idx_f, idx_s);
		break;
	default:
		_cm2_rps_empty(dst, src, idx_f, idx_s);
		break;
	}
}

// Internal function of CuteMatrix2 Library
// Vector scale
// Designed with REF
// Raw actuation
void _cm2_vecscale_empty_ra(float* dst, const float* src, float scale, unsigned width)
{
	float sv = (scale);
	const float* rb = src;
	float* rc = dst;
	float* rce = rc + width;
	for (; rc < rce; rc++, rb++)
	{
		float cv = *(rc);
		float bv = *(rb);
		cv = _cs2_fmadd(sv, bv, cv);
		_cs2_store(rc, cv);
	}
}

// Internal function of CuteMatrix2 Library
// Vector scale
// Designed with AVX
// Raw actuation
void _cm2_vecscale_avx_ra(float* dst, const float* src, float scale, unsigned width)
{
	__m128 sv = _mm_set_ps1(scale);
	const float* rb = src;
	float* rc = dst;
	float* rce = rc + width;
	for (; rc < rce; rc += 4, rb += 4)
	{
		__m128 cv = _mm_loadu_ps(rc);
		__m128 bv = _mm_loadu_ps(rb);
		cv = _mm_fmadd_ps(sv, bv, cv);
		_mm_storeu_ps(rc, cv);
	}
}

// Internal function of CuteMatrix2 Library
// Vector scale
// Designed with AVX2
// Raw actuation
void _cm2_vecscale_avx2_ra(float* dst, const float* src, float scale, unsigned width)
{
	__m256 sv = _mm256_set1_ps(scale);
	const float* rb = src;
	float* rc = dst;
	float* rce = rc + width;
	for (; rc < rce; rc += 8, rb += 8)
	{
		__m256 cv = _mm256_loadu_ps(rc);
		__m256 bv = _mm256_loadu_ps(rb);
		cv = _mm256_fmadd_ps(sv, bv, cv);
		_mm256_storeu_ps(rc, cv);
	}
}

template <typename _Func_type, typename _Cm2x>
void _cm2x_mul_range(
	_Func_type cm2x_vecscale_x,
	_Cm2x* dst,
	const _Cm2x* src_a,
	const _Cm2x* src_b,
	unsigned idx_begin,
	unsigned idx_end,
	unsigned width,
	unsigned size
)
{
	for (int i = idx_begin; i < idx_end; i++)
	{
		for (int k = 0; k < width; k++)
		{
			cm2x_vecscale_x(
				&dst->data[i * width],
				&src_b->data[k * width],
				src_a->data[i * width + k],
				size
			);
		}
	}
}

void _cm2_mul_empty(_cm2* dst, const _cm2* src_a, const _cm2* src_b)
{
	//cm2x_mul_range(cm2_vecscale_ps_empty_ra, dst, src_b, src_a, 0, 3,3, 3);
	_cm2x_mul_range(_cm2_vecscale_empty_ra, dst, src_b, src_a, 0, src_a->height, src_b->block_width, src_a->width);
}

void _cm2_mul_avx(_cm2* dst, const _cm2* src_a, const _cm2* src_b)
{
	_cm2x_mul_range(_cm2_vecscale_avx_ra, dst, src_b, src_a, 0, dst->width, dst->width, dst->height);
}

void _cm2_mul_avx2(_cm2* dst, const _cm2* src_a, const _cm2* src_b)
{
	_cm2x_mul_range(_cm2_vecscale_avx2_ra, dst, src_b, src_a, 0, dst->width, dst->width, dst->height);
}

void cm2_mul(_cm2* dst, const _cm2* src_a, const _cm2* src_b)
{
	switch (cm2_simd_support)
	{
	case CM2_SIMD_SUPPORT_EMPTY:
		_cm2_mul_empty(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX:
		_cm2_mul_avx(dst, src_a, src_b);
		break;
	case CM2_SIMD_SUPPORT_AVX2:
		_cm2_mul_avx2(dst, src_a, src_b);
		break;
	default:
		_cm2_mul_avx2(dst, src_a, src_b);
		break;
	}
}

void cm2_set_simd_support(_cm2_simd_support simd_support)
{
	cm2_simd_support = simd_support;
}

_cm2_simd_support cm2_get_simd_support()
{
	return cm2_simd_support;
}


_cm2::_cm2()
{
	this->width = 0u;
	this->height = 0u;
	this->block_width = 0u;
	this->block_height = 0u;
	this->byte_size = 0u;
	this->byte_width = 0u;
	this->byte_height = 0u;
	this->size = 0u;
	this->physical_size = 0u;
	this->blocks_count = 0u;
	this->data = nullptr;
	this->data_ptr_end = nullptr;
}

_cm2::_cm2(unsigned int width, unsigned int height)
{
	this->width = width;
	this->height = height;
	this->block_width = _cm2_get_block_size(this->width, CM2_BLOCK_SIZE);
	this->block_height = _cm2_get_block_size(this->height, CM2_BLOCK_SIZE);
	this->byte_size = _cm2_get_byte_size(this);
	this->byte_width = this->width * sizeof(float);
	this->byte_height = this->height * sizeof(float);
	this->size = this->width * this->height;
	this->physical_size = this->block_width * this->block_height;
	this->blocks_count = this->size / CM2_BLOCK_SIZE;
	this->data = new float[this->physical_size] {};
	this->data_ptr_end = this->data + this->physical_size - (this->block_height - this->height) * this->block_width;
}

_cm2::_cm2(unsigned int width, unsigned int height, const float* data)
{
	this->width = width;
	this->height = height;
	this->block_width = _cm2_get_block_size(this->width, CM2_BLOCK_SIZE);
	this->block_height = _cm2_get_block_size(this->height, CM2_BLOCK_SIZE);
	this->byte_size = _cm2_get_byte_size(this);
	this->byte_width = this->width * sizeof(float);
	this->byte_height = this->height * sizeof(float);
	this->size = this->width * this->height;
	this->physical_size = this->block_width * this->block_height;
	this->blocks_count = this->size / CM2_BLOCK_SIZE;
	this->data = new float[this->physical_size] {};
	this->data_ptr_end = this->data + this->physical_size - (this->block_height - this->height) * this->block_width;
	cm2_load(this, data);
}

// inline _cm2::~_cm2()
// {
// 	delete[] this->data;
// }

_cv2::_cv2()
{
	this->width = 0u;
	this->block_width = 0u;
	this->byte_size = 0u;
	this->byte_width = 0u;
	this->size = 0u;
	this->physical_size = 0u;
	this->blocks_count = 0u;
	this->data = nullptr;
	this->data_ptr_end = nullptr;
}

_cv2::_cv2(unsigned int width)
{
	this->width = width;
	this->block_width = _cm2_get_block_size(this->width, CM2_BLOCK_SIZE);
	this->byte_size = _cv2_get_byte_size(this);
	this->byte_width = this->width * sizeof(float);
	this->size = this->width;
	this->physical_size = this->block_width;
	this->blocks_count = this->size / CM2_BLOCK_SIZE;
	this->data = new float[this->physical_size] {};
	this->data_ptr_end = this->data + this->width;
}

_cv2::_cv2(unsigned int width, const float* data)
{
	this->width = width;
	this->block_width = _cm2_get_block_size(this->width, CM2_BLOCK_SIZE);
	this->byte_size = _cv2_get_byte_size(this);
	this->byte_width = this->width * sizeof(float);
	this->size = this->width;
	this->physical_size = this->block_width;
	this->blocks_count = this->size / CM2_BLOCK_SIZE;
	this->data = new float[this->physical_size] {};
	this->data_ptr_end = this->data + this->width;
	cv2_load(this, data);
}

// inline _cv2::~_cv2()
// {
// 	delete[] this->data;
// }

enum class mvc_e : int
{
	matrix = 0,
	vector = 1,
	constant = 2,
	incorrect = 3
};

class mvc_t
{
public:
	mvc_t() = default;

	mvc_t(const mvc_t& other)
	{
		//this->operator=(other);

		_mvc_e = other._mvc_e;
		_value = other._value;
	}

	mvc_t(std::shared_ptr<_cm2> other)
	{
		_mvc_e = mvc_e::matrix;
		_value = other;
	}

	mvc_t(std::shared_ptr<_cv2> other)
	{
		_mvc_e = mvc_e::vector;
		_value = other;
	}

	mvc_t(const _cm2* other)
	{
		this->operator=(other);
	}

	mvc_t(const _cv2* other)
	{
		this->operator=(other);
	}

	mvc_t(float other)
	{
		*this = other;
	}

	mvc_t(double other)
	{
		*this = other;
	}

	mvc_t(int other)
	{
		*this = other;
	}

	mvc_t& operator = (mvc_t other)
	{
		if (this == &other)
			return *this;
		_mvc_e = other._mvc_e;
		_value = other._value;
		return *this;
	}

	mvc_t& operator = (_cm2* other)
	{
		_mvc_e = mvc_e::matrix;
		_value = std::make_shared<_cm2>(other->width, other->height, other->data);
		return *this;
	}

	mvc_t& operator = (_cv2* other)
	{
		_mvc_e = mvc_e::vector;
		_value = std::make_shared<_cv2>(other->width, other->data);
		return *this;
	}

	mvc_t& operator = (float other)
	{
		_mvc_e = mvc_e::constant;
		_value = std::make_shared<float>(other);
		return *this;
	}

	mvc_t& operator = (double other)
	{
		_mvc_e = mvc_e::constant;
		_value = std::make_shared<float>(static_cast<float>(other));
		return *this;
	}

	mvc_t& operator = (int other)
	{
		_mvc_e = mvc_e::constant;
		_value = std::make_shared<float>(static_cast<float>(other));
		return *this;
	}

	mvc_t operator + (mvc_t other)
	{
		mvc_t mvc;
		if (this == &other)
		{
			return *this;
		}
		if (_mvc_e == mvc_e::constant && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::constant;
			const auto& lhs = std::any_cast<std::shared_ptr<float>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			mvc._value = std::make_shared<float>(*lhs + *rhs);
			return mvc;
		}
		if (_mvc_e == mvc_e::vector && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::vector;
			const auto& lhs = std::any_cast<std::shared_ptr<_cv2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cv2>(lhs->width);
			cv2_add(dst.get(), lhs.get(), *rhs);
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::vector && other._mvc_e == mvc_e::vector)
		{
			mvc._mvc_e = mvc_e::vector;
			const auto& lhs = std::any_cast<std::shared_ptr<_cv2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<_cv2>>(other._value);
			if (!(lhs->width == rhs->width))
			{
				throw std::invalid_argument(_CM2MM_MVC_ERROR_INCORRECT_MATRIX_SIZE);
			}
			const auto& dst = std::make_shared<_cv2>(lhs->width);
			cv2_add(dst.get(), lhs.get(), rhs.get());
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_add(dst.get(), lhs.get(), *rhs);
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::matrix)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<_cm2>>(other._value);
			if (!(lhs->width == rhs->width && lhs->height == rhs->height))
			{
				throw std::invalid_argument(_CM2MM_MVC_ERROR_INCORRECT_MATRIX_SIZE);
			}
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_add(dst.get(), lhs.get(), rhs.get());
			mvc._value = dst;
			return mvc;
		}
		throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
		return *this;
	}

	mvc_t operator - (mvc_t other)
	{
		mvc_t mvc;
		if (this == &other)
		{
			return *this;
		}
		if (_mvc_e == mvc_e::constant && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::constant;
			const auto& lhs = std::any_cast<std::shared_ptr<float>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			mvc._value = std::make_shared<float>(*lhs - *rhs);
			return mvc;
		}
		if (_mvc_e == mvc_e::vector && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::vector;
			const auto& lhs = std::any_cast<std::shared_ptr<_cv2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cv2>(lhs->width);
			cv2_sub(dst.get(), lhs.get(), *rhs);
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::vector && other._mvc_e == mvc_e::vector)
		{
			mvc._mvc_e = mvc_e::vector;
			const auto& lhs = std::any_cast<std::shared_ptr<_cv2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<_cv2>>(other._value);
			if (!(lhs->width == rhs->width))
			{
				return *this;
			}
			const auto& dst = std::make_shared<_cv2>(lhs->width);
			cv2_sub(dst.get(), lhs.get(), rhs.get());
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_sub(dst.get(), lhs.get(), *rhs);
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::matrix)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<_cm2>>(other._value);
			if (!(lhs->width == rhs->width && lhs->height == rhs->height))
			{
				throw std::invalid_argument(_CM2MM_MVC_ERROR_INCORRECT_MATRIX_SIZE);
			}
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_sub(dst.get(), lhs.get(), rhs.get());
			mvc._value = dst;
			return mvc;
		}
		throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
		return *this;
	}

	mvc_t operator * (mvc_t other)
	{
		mvc_t mvc;
		if (this == &other)
		{
			return *this;
		}
		if (_mvc_e == mvc_e::constant && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::constant;
			const auto& lhs = std::any_cast<std::shared_ptr<float>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			mvc._value = std::make_shared<float>(*lhs * *rhs);
			return mvc;
		}
		if (_mvc_e == mvc_e::vector && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::vector;
			const auto& lhs = std::any_cast<std::shared_ptr<_cv2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cv2>(lhs->width);
			cv2_scl(dst.get(), lhs.get(), *rhs);
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::matrix)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<_cm2>>(other._value);
			if (!(lhs->width == rhs->width && lhs->height == rhs->height))
			{
				throw std::invalid_argument(_CM2MM_MVC_ERROR_INCORRECT_MATRIX_SIZE);
			}
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_mul(dst.get(), lhs.get(), rhs.get());
			mvc._value = dst;
			return mvc;
		}
		if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_scl(dst.get(), lhs.get(), *rhs);
			mvc._value = dst;
			return mvc;
		}
		throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
		return *this;
	}

	mvc_t operator / (mvc_t other)
	{
		mvc_t mvc;
		if (this == &other)
		{
			return *this;
		}
		if (_mvc_e == mvc_e::constant && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::constant;
			const auto& lhs = std::any_cast<std::shared_ptr<float>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			mvc._value = std::make_shared<float>(*lhs / *rhs);
			return mvc;
		}
		if (_mvc_e == mvc_e::vector && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::vector;
			const auto& lhs = std::any_cast<std::shared_ptr<_cv2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cv2>(lhs->width);
			cv2_scl(dst.get(), lhs.get(), 1.f / *rhs);
			mvc._value = dst;
			return mvc;
		}
		else if (_mvc_e == mvc_e::matrix && other._mvc_e == mvc_e::constant)
		{
			mvc._mvc_e = mvc_e::matrix;
			const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_value);
			const auto& rhs = std::any_cast<std::shared_ptr<float>>(other._value);
			const auto& dst = std::make_shared<_cm2>(lhs->width, lhs->height);
			cm2_scl(dst.get(), lhs.get(), 1.f / *rhs);
			mvc._value = dst;
			return mvc;
		}
		return *this;
	}

	mvc_e get_type() const
	{
		return _mvc_e;
	}

	template <typename _Value_type>
	_Value_type get_value() const
	{
		const auto& value = std::any_cast<std::shared_ptr<_Value_type>>(_value);
		return *value;
	}

	template <typename _Value_type>
	_Value_type get_value_raw() const
	{
		const auto& value = std::any_cast<_Value_type>(_value);
		return value;
	}


private:
	mvc_e _mvc_e = mvc_e::incorrect;
	std::any _value;
	friend mvc_t _cm2mm_pow(mvc_t _Xx, mvc_t _Yx);
	friend mvc_t _cm2mm_stp(mvc_t _Xx, mvc_t _Yx);
	friend mvc_t _cm2mm_tsp(mvc_t _Xx);
	friend mvc_t _cm2mm_inv(mvc_t _Xx);
	friend mvc_t _cm2mm_rps(mvc_t _Xx, mvc_t _Yx, mvc_t _Zx);
	friend mvc_t _cm2mm_cos(mvc_t _Xx);
	friend mvc_t _cm2mm_sin(mvc_t _Xx);
	friend mvc_t _cm2mm_tan(mvc_t _Xx);
};

mvc_t _cm2mm_pow(mvc_t _Xx, mvc_t _Yx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::constant && _Yx._mvc_e == mvc_e::constant)
	{
		mvc._mvc_e = mvc_e::constant;
		const auto& lhs = std::any_cast<std::shared_ptr<float>>(_Xx._value);
		const auto& rhs = std::any_cast<std::shared_ptr<float>>(_Yx._value);
		mvc._value = std::make_shared<float>(std::pow(*lhs, *rhs));
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
	return mvc;
}

mvc_t _cm2mm_cos(mvc_t _Xx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::constant)
	{
		mvc._mvc_e = mvc_e::constant;
		const auto& arg = std::any_cast<std::shared_ptr<float>>(_Xx._value);
		mvc._value = std::make_shared<float>(std::cos(*arg));
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::vector)
	{
		mvc._mvc_e = mvc_e::vector;
		const auto& src = std::any_cast<std::shared_ptr<_cv2>>(_Xx._value);
		std::shared_ptr<_cv2> dst = std::make_shared<_cv2>(src->width);
		cv2_cos(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::matrix)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& src = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(src->width, src->height);
		cm2_cos(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}


mvc_t _cm2mm_sin(mvc_t _Xx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::constant)
	{
		mvc._mvc_e = mvc_e::constant;
		const auto& arg = std::any_cast<std::shared_ptr<float>>(_Xx._value);
		mvc._value = std::make_shared<float>(std::sin(*arg));
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::vector)
	{
		mvc._mvc_e = mvc_e::vector;
		const auto& src = std::any_cast<std::shared_ptr<_cv2>>(_Xx._value);
		std::shared_ptr<_cv2> dst = std::make_shared<_cv2>(src->width);
		cv2_sin(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::matrix)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& src = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(src->width, src->height);
		cm2_sin(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}

mvc_t _cm2mm_tan(mvc_t _Xx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::constant)
	{
		mvc._mvc_e = mvc_e::constant;
		const auto& arg = std::any_cast<std::shared_ptr<float>>(_Xx._value);
		mvc._value = std::make_shared<float>(std::sin(*arg));
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::vector)
	{
		mvc._mvc_e = mvc_e::vector;
		const auto& src = std::any_cast<std::shared_ptr<_cv2>>(_Xx._value);
		std::shared_ptr<_cv2> dst = std::make_shared<_cv2>(src->width);
		cv2_tan(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::matrix)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& src = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(src->width, src->height);
		cm2_tan(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}

mvc_t _cm2mm_stp(mvc_t _Xx, mvc_t _Yx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::matrix && _Yx._mvc_e == mvc_e::vector)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& lhs = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		const auto& rhs = std::any_cast<std::shared_ptr<_cv2>>(_Yx._value);
		if (!(lhs->width == rhs->width))
		{
			throw std::invalid_argument(_CM2MM_MVC_ERROR_VECTOR_HAVE_INCORRECT_SIZE);
		}
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(lhs->width, lhs->height);
		cm2_stp(dst.get(), lhs.get(), rhs.get());
		//_cm2_set11_ps(dst.get(), rhs->data[3]);
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}

mvc_t _cm2mm_tsp(mvc_t _Xx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::matrix)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& src = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(src->height, src->width);
		cm2_tsp(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}

mvc_t _cm2mm_rps(mvc_t _Xx, mvc_t _Yx, mvc_t _Zx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::vector && _Yx._mvc_e == mvc_e::constant && _Zx._mvc_e == mvc_e::constant)
	{
		mvc._mvc_e = mvc_e::vector;
		const auto& fhs = std::any_cast<std::shared_ptr<_cv2>>(_Xx._value);
		const auto& shs = std::any_cast<std::shared_ptr<float>>(_Yx._value);
		const auto& ths = std::any_cast<std::shared_ptr<float>>(_Zx._value);
		std::shared_ptr<_cv2> dst = std::make_shared<_cv2>(fhs->width);
		cv2_rps(dst.get(), fhs.get(), *shs, *ths);
		mvc._value = dst;
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::matrix && _Yx._mvc_e == mvc_e::constant && _Zx._mvc_e == mvc_e::constant)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& fhs = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		const auto& shs = std::any_cast<std::shared_ptr<float>>(_Yx._value);
		const auto& ths = std::any_cast<std::shared_ptr<float>>(_Zx._value);
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(fhs->width, fhs->height);
		cm2_rps(dst.get(), fhs.get(), *shs, *ths);
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}

mvc_t _cm2mm_inv(mvc_t _Xx)
{
	mvc_t mvc;
	if (_Xx._mvc_e == mvc_e::vector)
	{
		mvc._mvc_e = mvc_e::vector;
		const auto& src = std::any_cast<std::shared_ptr<_cv2>>(_Xx._value);
		std::shared_ptr<_cv2> dst = std::make_shared<_cv2>(src->width);
		cv2_inv(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	if (_Xx._mvc_e == mvc_e::matrix)
	{
		mvc._mvc_e = mvc_e::matrix;
		const auto& src = std::any_cast<std::shared_ptr<_cm2>>(_Xx._value);
		std::shared_ptr<_cm2> dst = std::make_shared<_cm2>(src->width, src->height);
		cm2_inv(dst.get(), src.get());
		mvc._value = dst;
		return mvc;
	}
	throw std::invalid_argument(_CM2MM_MVC_ERROR_USUPORTED_OPERATION);
}

static std::map<std::string, mvc_t> _cm2mm_mvc_map;

static std::map<std::string, std::shared_ptr<std::string>> _cm2mm_constant_map;

static std::map<std::string, std::shared_ptr<std::string>> _cm2mm_vector_map;

static std::map<std::string, std::shared_ptr<std::string>> _cm2mm_matrix_map;

class Expression {
public:
	Expression(const std::string& str)
	{
		m_expression = str;
		const auto& remove_if_result = std::remove_if(m_expression.begin(), m_expression.end(), isspace);
		m_expression.erase(remove_if_result, m_expression.end());
	}

	mvc_t evaluate()
	{
		m_position = 0;
		mvc_t mvc_expr_result = parseExpression();
		unsigned expr_size = m_expression.size();
		if (m_position != expr_size)
		{
			throw std::invalid_argument(_CM2MM_MVC_ERROR_INVALID_EXPRESSION);
		}
		return mvc_expr_result;
	}

	static void setVariable(const std::string& variable, mvc_t value)
	{
		_cm2mm_mvc_map[variable] = value;
	}

private:
	mvc_t parseExpression()
	{
		return parseTerm();
	}
	mvc_t parseTerm()
	{
		mvc_t left = parseFactor();
		while (true) {
			if (m_expression[m_position] == '+') {
				m_position++;
				left = left + parseFactor();
			}
			else if (m_expression[m_position] == '-') {
				m_position++;
				left = left - parseFactor();
			}
			else {
				break;
			}
		}
		return left;
	}
	mvc_t parseFactor()
	{
		mvc_t left = parsePow();
		while (true)
		{
			if (m_expression[m_position] == '*') {
				m_position++;
				left = left * parsePow();
			}
			else if (m_expression[m_position] == '/') {
				m_position++;
				left = left / parsePow();
			}
			else {
				break;
			}
		}
		return left;
	}

	mvc_t parsePow()
	{
		mvc_t left = parsePrimary();
		while (true)
		{
			if (m_expression[m_position] == '^') {
				m_position++;
				left = _cm2mm_pow(left, parsePrimary());
			}
			else {
				break;
			}
		}
		return left;
	}

	mvc_t parsePrimary()
	{
		if (m_expression[m_position] == '(') {
			m_position++;
			mvc_t value = parseExpression();
			m_position++; // пропускаем закрывающую скобку
			return value;
		}
		else if (std::isdigit(m_expression[m_position]))
		{
			std::string number;
			while (std::isdigit(m_expression[m_position]) || m_expression[m_position] == '.')
			{
				number += m_expression[m_position];
				m_position++;
			}
			mvc_t parse_value;
			parse_value = std::strtod(number.c_str(), nullptr);
			return parse_value;
		}
		else if (m_expression.substr(m_position, 3) == "stp")
		{
			m_position += 3; // пропускаем "stp"
			m_position++; // пропускаем открывающую скобку
			mvc_t valuelhs = parseExpression();
			m_position++; // пропускаем запятую
			mvc_t valuerhs = parseExpression();
			m_position++; // пропускаем закрывающую скобку
			return _cm2mm_stp(valuelhs, valuerhs);
		}
		else if (m_expression.substr(m_position, 3) == "tsp")
		{
			m_position += 3;
			m_position++;
			mvc_t value = parseExpression();
			m_position++;
			return _cm2mm_tsp(value);
		}
		else if (m_expression.substr(m_position, 3) == "inv")
		{
			m_position += 3;
			m_position++;
			mvc_t value = parseExpression();
			m_position++;
			return _cm2mm_inv(value);
		}
		else if (m_expression.substr(m_position, 3) == "rps")
		{
			m_position += 3;
			m_position++;
			mvc_t valuefhs = parseExpression();
			m_position++;
			mvc_t valueshs = parseExpression();
			m_position++;
			mvc_t valueths = parseExpression();
			m_position++;
			return _cm2mm_rps(valuefhs, valueshs, valueths);
		}
		else if (m_expression.substr(m_position, 3) == "cos")
		{
			m_position += 3;
			m_position++;
			mvc_t value = parseExpression();
			m_position++;
			return _cm2mm_cos(value);
		}
		else if (m_expression.substr(m_position, 3) == "sin")
		{
			m_position += 3;
			m_position++;
			mvc_t value = parseExpression();
			m_position++;
			return _cm2mm_sin(value);
		}
		else if (m_expression.substr(m_position, 3) == "tan")
		{
			m_position += 3;
			m_position++;
			mvc_t value = parseExpression();
			m_position++;
			return _cm2mm_tan(value);
		}
		else if (m_expression[m_position] == '-')
		{
			m_position++;
			return parsePrimary() * mvc_t(-1);
		}
		else if (std::isalpha(m_expression[m_position]) || m_expression[m_position] == '_')
		{ // новое условие
			std::string variable;
			while (std::isalpha(m_expression[m_position]) || std::isdigit(m_expression[m_position]) || m_expression[m_position] == '_')
			{
				variable += m_expression[m_position];
				m_position++;
			}
			auto it = _cm2mm_mvc_map.find(variable);
			if (it != _cm2mm_mvc_map.end())
			{
				return it->second;
			}
			else
			{
				std::string exception_str = _CM2MM_MVC_ERROR_INVALID_EXPRESSION_UNKNOWN_VAR_FUNC(variable);
				throw std::invalid_argument(exception_str);
			}
		}
		else
		{
			throw std::invalid_argument(_CM2MM_MVC_ERROR_INVALID_EXPRESSION);
		}
	}

	std::string m_expression;
	size_t m_position;
};

template<class _Stl_Stream = std::stringstream>
inline void _cm2mm_parse_matrix(const std::string& s_dst, const std::string& value, std::shared_ptr<std::string> buffer = nullptr)
{
	std::shared_ptr<_cm2> p_dst;

	_Stl_Stream s_stream(value);

	unsigned width, height;
	if (buffer.get() != nullptr)
	{
		std::string width_to_write_str;
		std::string height_to_write_str;
		s_stream >> width_to_write_str >> height_to_write_str;
		*buffer = width_to_write_str + " " + height_to_write_str + " ";
		width = std::stof(width_to_write_str);
		height = std::stof(height_to_write_str);
	}
	else
	{
		s_stream >> width >> height;
	}
	p_dst = std::make_shared<_cm2>(width, height);

	if (buffer.get() != nullptr)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				std::string value_to_write_str;
				s_stream >> value_to_write_str;
				*buffer += value_to_write_str + " ";
				float value_to_write = std::stof(value_to_write_str);
				p_dst->data[y * width + x] = value_to_write;
			}
		}
	}
	else
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float value_to_write = 0;
				s_stream >> value_to_write;
				p_dst->data[y * width + x] = value_to_write;
			}
		}
	}

	Expression::setVariable(s_dst, p_dst);
}

template<class _Stl_Stream = std::stringstream>
inline void _cm2mm_parse_vector(const std::string& s_dst, const std::string& value, std::shared_ptr<std::string> buffer = nullptr)
{
	std::shared_ptr<_cv2> p_dst;

	_Stl_Stream s_stream(value);

	unsigned width;
	if (buffer.get() != nullptr)
	{
		std::string width_to_write_str;
		s_stream >> width_to_write_str;
		*buffer = width_to_write_str + " ";
		width = std::stof(width_to_write_str);
	}
	else
	{
		s_stream >> width;
	}
	p_dst = std::make_shared<_cv2>(width);

	if (buffer.get() != nullptr)
	{
		for (int x = 0; x < width; x++)
		{
			std::string value_to_write_str;
			s_stream >> value_to_write_str;
			*buffer += value_to_write_str + " ";
			float value_to_write = std::stof(value_to_write_str);
			p_dst->data[x] = value_to_write;
		}
	}
	else
	{
		for (int x = 0; x < width; x++)
		{
			s_stream >> p_dst->data[x];
		}
	}

	Expression::setVariable(s_dst, p_dst);
}

template<class _Stl_Stream = std::stringstream>
inline void _cm2mm_parse_constant(const std::string& s_dst, const std::string& value, std::shared_ptr<std::string> buffer = nullptr)
{
	_Stl_Stream s_stream(value);

	float constant_value;

	if (buffer.get() != nullptr)
	{
		std::string value_to_write_str;
		s_stream >> value_to_write_str;
		*buffer = value_to_write_str;
		constant_value = std::stof(value_to_write_str);
		Expression::setVariable(s_dst, constant_value);
	}
	else
	{
		s_stream >> constant_value;
		Expression::setVariable(s_dst, constant_value);
	}

}

CM2MMAPI
void cm2mm_add_chars_matrix(const char* matrix_name, const char* chars) noexcept
{
	std::string s_matrix_name(matrix_name);
	std::string s_chars(chars);
	_cm2mm_parse_matrix(s_matrix_name, s_chars);
	_cm2mm_matrix_map[s_matrix_name] = std::make_shared<std::string>(s_chars);
}

CM2MMAPI
void cm2mm_add_file_matrix(const char* matrix_name, const char* path) noexcept
{
	std::string s_matrix_name(matrix_name);
	std::string s_path(path);
	auto& buffer_ptr = _cm2mm_matrix_map[s_matrix_name];
	buffer_ptr = std::make_shared<std::string>();
	_cm2mm_parse_matrix<std::ifstream>(s_matrix_name, s_path, buffer_ptr);
}

CM2MMAPI
void cm2mm_add_chars_vector(const char* vector_name, const char* chars) noexcept
{
	std::string s_vector_name(vector_name);
	std::string s_chars(chars);
	_cm2mm_parse_vector(s_vector_name, s_chars);
	_cm2mm_vector_map[s_vector_name] = std::make_shared<std::string>(s_chars);
}

CM2MMAPI
void cm2mm_add_file_vector(const char* vector_name, const char* path) noexcept
{
	std::string s_vector_name(vector_name);
	std::string s_path(path);
	auto& buffer_ptr = _cm2mm_vector_map[s_vector_name];
	buffer_ptr = std::make_shared<std::string>();
	_cm2mm_parse_vector<std::ifstream>(s_vector_name, s_path, buffer_ptr);
}

CM2MMAPI
void cm2mm_add_chars_constant(const char* constant_name, const char* chars) noexcept
{
	std::string s_constant_name(constant_name);
	std::string s_chars(chars);
	_cm2mm_parse_constant(s_constant_name, s_chars);
	_cm2mm_constant_map[s_constant_name] = std::make_shared<std::string>(s_chars);
}

CM2MMAPI
void cm2mm_add_file_constant(const char* constant_name, const char* path) noexcept
{
	std::string s_constant_name(constant_name);
	std::string s_path(path);
	auto& buffer_ptr = _cm2mm_constant_map[s_constant_name];
	buffer_ptr = std::make_shared<std::string>();
	_cm2mm_parse_constant<std::ifstream>(s_constant_name, s_path, buffer_ptr);
}

CM2MMAPI
unsigned cm2mm_get_width_of_matrix(const char* matrix_name) noexcept
{
	std::string s_matrix_name(matrix_name);
	const auto& cm2mvc = _cm2mm_mvc_map[s_matrix_name];
	const auto& cm2 = cm2mvc.get_value_raw<std::shared_ptr<_cm2>>();
	if (cm2mvc.get_type() != mvc_e::matrix) return 0;
	unsigned width_of_matrix = cm2->width;
	return width_of_matrix;
}

CM2MMAPI
unsigned cm2mm_get_height_of_matrix(const char* matrix_name) noexcept
{
	std::string s_matrix_name(matrix_name);
	const auto& cm2mvc = _cm2mm_mvc_map[s_matrix_name];
	const auto& cm2 = cm2mvc.get_value_raw<std::shared_ptr<_cm2>>();
	if (cm2mvc.get_type() != mvc_e::matrix) return 0;
	unsigned height_of_matrix = cm2->height;
	return height_of_matrix;
}

CM2MMAPI
unsigned cm2mm_get_size_of_matrix(const char* matrix_name) noexcept
{
	std::string s_matrix_name(matrix_name);
	const auto& cm2mvc = _cm2mm_mvc_map[s_matrix_name];
	const auto& cm2 = cm2mvc.get_value_raw<std::shared_ptr<_cm2>>();
	if (cm2mvc.get_type() != mvc_e::matrix) return 0;
	unsigned size_of_matrix = cm2->size;
	return size_of_matrix;
}

CM2MMAPI
unsigned cm2mm_get_size_of_vector(const char* vector_name) noexcept
{
	std::string s_vector_name(vector_name);
	const auto& cv2mvc = _cm2mm_mvc_map[s_vector_name];
	if (cv2mvc.get_type() != mvc_e::vector) return 0;
	const auto& cv2 = cv2mvc.get_value_raw<std::shared_ptr<_cv2>>();
	unsigned size_of_matrix = cv2->size;
	return size_of_matrix;
}

CM2MMAPI
const float* cm2mm_get_matrix(const char* matrix_name) noexcept
{
	std::string s_matrix_name(matrix_name);
	const auto& cm2mvc = _cm2mm_mvc_map[s_matrix_name];
	if (cm2mvc.get_type() != mvc_e::matrix) return nullptr;
	const auto& cm2 = cm2mvc.get_value<_cm2>();
	float* cm2_data = cm2.data;
	return cm2_data;
}

CM2MMAPI
const float* cm2mm_get_vector(const char* vector_name) noexcept
{
	std::string s_vector_name(vector_name);
	const auto& cv2mvc = _cm2mm_mvc_map[s_vector_name];
	if (cv2mvc.get_type() != mvc_e::vector) return nullptr;
	const auto& cv2 = cv2mvc.get_value_raw<std::shared_ptr<_cv2>>();
	const float* cv2_data = cv2->data;
	return cv2_data;
}

CM2MMAPI
float cm2mm_get_constant(const char* constant_name) noexcept
{
	std::string s_constant_name(constant_name);
	const auto& constantmvc = _cm2mm_mvc_map[s_constant_name];
	if (constantmvc.get_type() != mvc_e::constant) return 0;
	const auto& constant = constantmvc.get_value_raw<std::shared_ptr<float>>();
	float constant_data = *constant;
	return constant_data;
}

CM2MMAPI
int cm2mm_get_mvc_type(const char* constant_name) noexcept
{
	std::string s_constant_name(constant_name);
	const auto& constantmvc = _cm2mm_mvc_map[s_constant_name];
	return static_cast<int>(constantmvc.get_type());
}

struct mvc_value
{
	mvc_value() = default;
	mvc_value(const char* name, const char* value) :
		name(name),
		value(value)
	{}
	const char* name;
	const char* value;
};

CM2MMAPI
int cm2mm_get_constants_count() noexcept
{
	int constants_count = _cm2mm_constant_map.size();
	return constants_count;
}

CM2MMAPI
int cm2mm_get_vectors_count() noexcept
{
	int vectors_count = _cm2mm_vector_map.size();
	return vectors_count;
}

CM2MMAPI
int cm2mm_get_matrices_count() noexcept
{
	int matrices_count = _cm2mm_matrix_map.size();
	return matrices_count;
}

static std::vector<mvc_value> _cm2mm_mvc_value_vector;

static const mvc_value* _cm2mm_get_mvc_values(std::map<std::string, std::shared_ptr<std::string>>& mvc_value_map)
{
	_cm2mm_mvc_value_vector.clear();
	const unsigned reserve_size = mvc_value_map.size();
	_cm2mm_mvc_value_vector.reserve(reserve_size);
	decltype(auto) constant_map_iter = mvc_value_map.rbegin();
	decltype(auto) constant_map_iter_end = mvc_value_map.rend();
	while (constant_map_iter != constant_map_iter_end)
	{
		const char* mvc_name = constant_map_iter->first.c_str();
		const char* mvc_value = constant_map_iter->second->c_str();
		_cm2mm_mvc_value_vector.emplace_back(mvc_name, mvc_value);
		constant_map_iter++;
	}
	return _cm2mm_mvc_value_vector.data();
}

CM2MMAPI
const mvc_value* cm2mm_get_constants() noexcept
{
	decltype(auto) mvc_value_constants = _cm2mm_get_mvc_values(_cm2mm_constant_map);
	return mvc_value_constants;
}

CM2MMAPI
const mvc_value* cm2mm_get_vectors() noexcept
{
	decltype(auto) mvc_value_vectors = _cm2mm_get_mvc_values(_cm2mm_vector_map);
	return mvc_value_vectors;
}

CM2MMAPI
const mvc_value* cm2mm_get_matrices() noexcept
{
	decltype(auto) mvc_value_matrices = _cm2mm_get_mvc_values(_cm2mm_matrix_map);
	return mvc_value_matrices;
}

void _cm2mm_save_mvc_to_local_file(
	const std::string& mvc_name, 
	const std::string& mvc_directory,
	std::map<std::string, std::shared_ptr<std::string>>& mvc_value_map
)
{
	_mkdir(std::string("cm2mm/" + mvc_directory).c_str());
	std::ofstream outfile("cm2mm/" + mvc_directory + "/" + mvc_name + ".txt");
	const auto& mvc_value = *mvc_value_map[mvc_name];
	outfile << mvc_value;
	outfile.close();
}

CM2MMAPI void cm2mm_save_mvc_to_local_file(const char* mvc_name) noexcept
{
	int mvc_type = cm2mm_get_mvc_type(mvc_name);
	std::string s_name = std::string(mvc_name);
	_mkdir("cm2mm");
	switch (mvc_type)
	{
	case 0:
	{
		_cm2mm_save_mvc_to_local_file(s_name, "matrices", _cm2mm_matrix_map);
		break;
	}
	case 1:
	{
		_cm2mm_save_mvc_to_local_file(s_name, "vectors", _cm2mm_vector_map);
		break;
	}
	case 2:
	{
		_cm2mm_save_mvc_to_local_file(s_name, "constants", _cm2mm_constant_map);
		break;
	}
	default:
		break;
	}
}

template <typename _Func_Add_file_MVC>
void _cm2mm_load_all_mvc_from_local_file(const std::string& path, _Func_Add_file_MVC& func_add_file_mvc)
{
	decltype(auto) dir_iter = std::filesystem::directory_iterator(path);
	for (const auto& entry : dir_iter)
	{
		auto p = entry.path().stem().string();
		auto e = entry.path().string();
		const char* mvc_name = p.c_str();
		const char* mvc_path = e.c_str();
		func_add_file_mvc(mvc_name, mvc_path);
	}
}

CM2MMAPI void cm2mm_load_all_mvc_from_local_file() noexcept
{
	_cm2mm_load_all_mvc_from_local_file("cm2mm/matrices", cm2mm_add_file_matrix);
	_cm2mm_load_all_mvc_from_local_file("cm2mm/vectors", cm2mm_add_file_vector);
	_cm2mm_load_all_mvc_from_local_file("cm2mm/constants", cm2mm_add_file_constant);
}

//matrix-vector-constant name
static std::string _cm2mm_mvc_name = _CM2MM_MVC_NAME_;

static std::string _cm2mm_mvc_error_def = _CM2MM_MVC_ERROR_DEF_;

static unsigned _cm2mm_mvc_added_in_system_count = 0;

static std::string _cm2mm_last_error = "";

static float _cm2mm_last_eval_time = 0;

CM2MMAPI
const char* cm2mm_get_last_error() noexcept
{
	return _cm2mm_last_error.data();
}

CM2MMAPI
float cm2mm_get_last_eval_time() noexcept
{
	return _cm2mm_last_eval_time;
}

CM2MMAPI
const char* cm2mm_eval_expression(const char* expression) noexcept
{
	std::string s_expression(expression);
	Expression e(s_expression);
	std::string mvc_name = _cm2mm_mvc_name + std::to_string(_cm2mm_mvc_added_in_system_count);
	mvc_t& mvc_result = _cm2mm_mvc_map[mvc_name];
	decltype(auto) mvc_result_iter = _cm2mm_mvc_map.find(mvc_name);
	try
	{
		const auto t1 = high_resolution_clock::now();
		mvc_result_iter->second = e.evaluate();
		const auto t2 = high_resolution_clock::now();
		const auto ms_int = duration_cast<std::chrono::nanoseconds>(t2 - t1);
		const float nanoseconds = static_cast<float>(ms_int.count());
		const float milliseconds = nanoseconds / 1000000.f;
		_cm2mm_last_eval_time = milliseconds;
	}
	catch (std::invalid_argument e)
	{
		_cm2mm_last_error = e.what();
		_cm2mm_last_eval_time = 0.f;
		return _cm2mm_mvc_error_def.c_str();
	}
	_cm2mm_mvc_added_in_system_count++;
	return mvc_result_iter->first.c_str();
}

CM2MMAPI void cm2mm_set_simd_support() noexcept
{
	cm2_set_simd_support(CM2_SIMD_SUPPORT_AVX2);
}
