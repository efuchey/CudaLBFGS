#include "CudaLBFGS/lbfgs.h"
#include "CudaLBFGS/error_checking.h"
#include "CudaLBFGS/timer.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>

using namespace std;

class cpu_sqtrack_chi2 : public cpu_cost_function
{
public:
	cpu_sqtrack_chi2(float x0, float y0, float tx, float ty)//, float pinv) 
	: cpu_cost_function(4)
	, m_x0(x0)
	, m_y0(y0)
	, m_tx(tx)
	, m_ty(ty) {}
//	, m_tx(pinv) {}

	void cpu_f(const size_t n_points, const floatdouble* h_drift, const floatdouble* h_res, 
			const floatdouble* h_p1x, const floatdouble* h_p1y, const floatdouble* h_p1z, 
			const floatdouble* h_deltapx, const floatdouble* h_deltapy, const floatdouble* h_deltapz, 
			floatdouble& h_chi2)
	{
		h_chi2 = 0.0;
		for(size_t i = 0; i<n_points; i++){
			const float &Den = sqrtf( h_deltapy[i] * h_deltapy[i] +  h_deltapx[i] * h_deltapx[i] + 2.0f * h_deltapx[i] * h_deltapy[i] * m_tx * m_ty );
			const float &dca = ( -1.0f * h_deltapy[i] * ( h_p1x[i] - m_x0 ) + h_deltapx[i] * ( h_p1y[i] - m_y0 ) - h_p1z[i] * ( m_tx * h_deltapy[i] - m_ty * h_deltapy[i] )  ) / Den;
			const float &value = ( h_drift[i] - dca ) * ( h_drift[i] - dca ) / h_res[i] / h_res[i];
			h_chi2=+value;
		}
	}

	void cpu_gradf(const size_t n_points, const floatdouble* h_drift, const floatdouble* h_res, 
			const floatdouble* h_p1x, const floatdouble* h_p1y, const floatdouble* h_p1z, 
			const floatdouble* h_deltapx, const floatdouble* h_deltapy, const floatdouble* h_deltapz, 
			floatdouble *h_grad)
	{
		for (size_t n = 0; n<m_numDimensions; n++)
		{
			h_grad[n] = 0.0;
		}
		for(size_t i = 0; i<n_points; i++){
			const float &Den = sqrtf( h_deltapy[i] * h_deltapy[i] +  h_deltapx[i] * h_deltapx[i] + 2.0f * h_deltapx[i] * h_deltapy[i] * m_tx * m_ty );
			const float &dca = ( -1.0f * h_deltapy[i] * ( h_p1x[i] - m_x0 ) + h_deltapx[i] * ( h_p1y[i] - m_y0 ) - h_p1z[i] * ( m_tx * h_deltapy[i] - m_ty * h_deltapy[i] )  ) / Den;
			const float &value = ( h_drift[i] - dca ) * ( h_drift[i] - dca ) / h_res[i] / h_res[i];
			
			h_grad[0]+= -2.0f * value * h_deltapy[i] / Den * value / h_res[i] / h_res[i];
			h_grad[1]+= 2.0f * value * h_deltapx[i] / Den * value / h_res[i] / h_res[i];
			h_grad[2]+= -2.0f * value * ( h_deltapy[i] * h_p1z[i] / Den - dca * m_ty * h_deltapx[i] * h_deltapy[i] / Den / Den ) * value / h_res[i] / h_res[i];	
			h_grad[3]+= 2.0f * value * ( h_deltapx[i] * h_p1z[i] / Den - dca * m_tx * h_deltapx[i] * h_deltapy[i] / Den / Den ) * value / h_res[i] / h_res[i];
		}
	}

	void cpu_f_gradf(const floatdouble* h_x, floatdouble *h_chi2, floatdouble *h_gradf)//added this function to make the compiler happy
	{
		const size_t n_points = sizeof(h_x)/sizeof(floatdouble)/8;
		const floatdouble** h_dataarrays;
		for(int i = 0; i<n_points; i++){
			h_dataarrays[i] = h_x+i*n_points;
		}
		cpu_f_gradf(n_points, h_dataarrays[0], h_dataarrays[1], 
			h_dataarrays[2], h_dataarrays[3], h_dataarrays[4], 
			h_dataarrays[5], h_dataarrays[6], h_dataarrays[7],  
			*h_chi2, h_gradf);
	}
		
	void cpu_f_gradf(const size_t n_points, const floatdouble* h_drift, const floatdouble* h_res, 
			const floatdouble* h_p1x, const floatdouble* h_p1y, const floatdouble* h_p1z, 
			const floatdouble* h_deltapx, const floatdouble* h_deltapy, const floatdouble* h_deltapz, 
			floatdouble& h_chi2, floatdouble *h_gradf)
	{
		cpu_f(n_points, h_drift, h_res, 
			h_p1x, h_p1y, h_p1z, 
			h_deltapx, h_deltapy, h_deltapz, 
			h_chi2);
		cpu_gradf(n_points, h_drift, h_res, 
			h_p1x, h_p1y, h_p1z, 
			h_deltapx, h_deltapy, h_deltapz, 
			h_gradf);
	}
private:
	float m_x0;
	float m_y0;
	float m_tx;
	float m_ty;
//	float m_pinv;
};

namespace gpu_sqtrack_chi2_d
{
	__device__ float m_x0;
	__device__ float m_y0;
	__device__ float m_tx;
	__device__ float m_ty;
	//__device__ float m_pinv;

	__global__ void kernel_combined_f_gradf(const int n_points, const float* d_drift, const float* d_res, 
		const float* d_p1x, const float* d_p1y, const float* d_p1z, 
		const float* d_deltapx, const float* d_deltapy, const float* d_deltapz, 
		float& d_chi2, float* d_grad)
	{
		d_chi2 = 0.0f;
		d_grad[0] = 0.0f;
		d_grad[1] = 0.0f;
		d_grad[2] = 0.0f;
		d_grad[3] = 0.0f;
		for(size_t i = 0; i<n_points; i++){
			const float &Den = sqrtf( d_deltapy[i] * d_deltapy[i] +  d_deltapx[i] * d_deltapx[i] + 2.0f * d_deltapx[i] * d_deltapy[i] * m_tx * m_ty );
			const float &dca = ( -1.0f * d_deltapy[i] * ( d_p1x[i] - m_x0 ) + d_deltapx[i] * ( d_p1y[i] - m_y0 ) - d_p1z[i] * ( m_tx * d_deltapy[i] - m_ty * d_deltapy[i] )  ) / Den;
			const float &value = ( d_drift[i] - dca ) * ( d_drift[i] - dca ) / d_res[i] / d_res[i];
			d_chi2=+value;
			
			d_grad[0]+= -2.0f * value * d_deltapy[i] / Den * value / d_res[i] / d_res[i];
			d_grad[1]+= 2.0f * value * d_deltapx[i] / Den * value / d_res[i] / d_res[i];
			d_grad[2]+= -2.0f * value * ( d_deltapy[i] * d_p1z[i] / Den - dca * m_ty * d_deltapx[i] * d_deltapy[i] / Den / Den ) * value / d_res[i] / d_res[i];	
			d_grad[3]+= 2.0f * value * ( d_deltapx[i] * d_p1z[i] / Den - dca * m_tx * d_deltapx[i] * d_deltapy[i] / Den /Den ) * value / d_res[i] / d_res[i];
		}
	}
}

class gpu_sqtrack_chi2 : public cost_function
{
public:
	gpu_sqtrack_chi2(float x0, float y0, float tx, float ty)//, float pinv 
	: cost_function(4) 
	{
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_x0, gpu_sqtrack_chi2_d::m_x0) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_y0, gpu_sqtrack_chi2_d::m_y0) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_tx, gpu_sqtrack_chi2_d::m_tx) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_ty, gpu_sqtrack_chi2_d::m_tx) );
//		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_pinv, gpu_sqtrack_chi2_d::m_pinv) );

		CudaSafeCall( cudaMemcpy(m_d_x0, &x0, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_y0, &y0, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_tx, &tx, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_ty, &ty, sizeof(float), cudaMemcpyHostToDevice) );
//		CudaSafeCall( cudaMemcpy(m_d_pinv, &pinv, sizeof(float), cudaMemcpyHostToDevice) );
	}
	
	void f_gradf(const float* h_x, float *h_chi2, float *h_gradf)//added this function to make the compiler happy
	{
		const size_t n_points = sizeof(h_x)/sizeof(float)/8;
		const float** h_dataarrays;
		for(int i = 0; i<n_points; i++){
			h_dataarrays[i] = h_x+i*n_points;
		}
		f_gradf(n_points, h_dataarrays[0], h_dataarrays[1], 
			h_dataarrays[2], h_dataarrays[3], h_dataarrays[4], 
			h_dataarrays[5], h_dataarrays[6], h_dataarrays[7],  
			*h_chi2, h_gradf);
	}
	
	void f_gradf(const int n_points, const float* d_drift, const float* d_res, 
			const float* d_p1x, const float* d_p1y, const float* d_p1z, 
			const float* d_deltapx, const float* d_deltapy, const float* d_deltapz, 
			float& d_chi2, float* d_grad)
	{
		gpu_sqtrack_chi2_d::kernel_combined_f_gradf<<<1, 1>>>(n_points, d_drift, d_res, 
			d_p1x, d_p1y, d_p1z, 
			d_deltapx, d_deltapy, d_deltapz, 
			d_chi2, d_grad);
		CudaCheckError();
	}
	
private:

	float *m_d_x0; 
	float *m_d_y0;
	float *m_d_tx;
	float *m_d_ty;
//	float *m_d_pinv;
};

int main(int argc, char **argv)
{
	// CPU
/*
	cpu_sqtrack_chi2 p1(4.0f, 2.0f, 6.0f);
	lbfgs minimizer1(p1);

	float x = 8.0f;
	{
		timer t("sqtrack_chi2_cpu");
		t.start();
		minimizer1.minimize_with_host_x(&x);
	}
	
	cout << "CPU sqtrack_Chi2: " << x << endl;

	// GPU

	gpu_sqtrack_chi2 p2(4.0f, 2.0f, 6.0f);
	lbfgs minimizer2(p2);
	
	x = 8.0f;
	
	float *d_x;
	CudaSafeCall( cudaMalloc(&d_x, sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_x, &x, sizeof(float), cudaMemcpyHostToDevice) );

	{
		timer t("sqtrack_chi2_gpu");
		t.start();
		minimizer2.minimize(d_x);
	}

	CudaSafeCall( cudaMemcpy(&x, d_x, sizeof(float), cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaFree(d_x) );

	cout << "GPU sqtrack_Chi2: " << x << endl;
*/
	return 0;
}

