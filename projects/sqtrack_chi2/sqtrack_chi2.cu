#include "CudaLBFGS/lbfgs.h"
#include "CudaLBFGS/error_checking.h"
#include "CudaLBFGS/timer.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

class cpu_sqtrack_chi2 : public cpu_cost_function
{
public:
	cpu_sqtrack_chi2(float a, float b, float c) 
	: cpu_cost_function(1)
	, m_a(a)
	, m_b(b)
	, m_c(c) {}
	
	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		const float x = *h_x;
		*h_y = m_a * x * x + m_b * x + m_c;
	}
	
	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		const float x = *h_x;
		*h_grad = 2.0f * m_a * x + m_b;
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
		cpu_f(h_x, h_f);
		cpu_gradf(h_x, h_gradf);
	}
	
private:
	float m_x0;
	float m_y0;
	float m_tx;
	float m_ty;
	float m_pinv;
};

namespace gpu_sqtrack_chi2_d
{
	__device__ float m_x0;
	__device__ float m_y0;
	__device__ float m_tx;
	__device__ float m_ty;
	__device__ float m_pinv;

	__global__ void kernel_combined_f_gradf(const float *d_drift, const float *d_res, 
		const float *d_p1x, const float *d_p1y, const float *d_p1z, 
		const float *d_deltapx, const float *d_deltapy, const float *d_deltapz, 
		float *d_chi2, float *d_grad)
	{

		const float &Den = sqrtf( *d_deltapy * *d_deltapy +  *d_deltapx * *d_deltapx + 2.0f * *d_deltapx * *d_deltapy * m_tx * m_ty );
		const float &dca = ( -1.0f * *d_deltapy * ( d_p1x - m_x0 ) + *d_deltapx * ( d_p1y - m_y0 ) - *d_p1z ( m_tx * *d_deltapy- m_ty * *d_deltapy )  ) / Den;
		const float &value = ( *d_drift - dca ) * ( *d_drift - dca ) / *d_res / *d_res;
		*d_chi2 = value;
/*
	for(int i = 0; i<n_points; i++){
		dca = ( -deltapy[i]*(p1x[i]-output_parameters[0]) + deltapx[i]*(p1y[i]-output_parameters[1]) + p1z[i]*(output_parameters[2]*deltapy[i]-output_parameters[3]*deltapx[i]) ) / sqrtf( deltapy[i]*deltapy[i] + deltapx[i]*deltapx[i] - 2*output_parameters[2]*output_parameters[3]*deltapy[i]*deltapx[i] );
		values[i] = (driftdist[i] - dca) * (driftdist[i] - dca) / resolutions[i] / resolutions[i];
		chi2+= values[i];

	     	Den2 = deltapy[i]*deltapy[i] + deltapx[i]*deltapx[i] - 2 * ( deltapx[i]*deltapy[i]*output_parameters[2]*output_parameters[3]);
	     	Den = sqrtf(Den2);
		
		//dchi2/dx0:
		derivatives[0*n_points+i] = -2*values[i]*deltapy[i]/Den;
		//dchi2/dy0:
		derivatives[1*n_points+i] = +2*values[i]*deltapx[i]/Den;
		//dchi2/dtx:
		derivatives[2*n_points+i] = -2*values[i]*(deltapy[i]*p1z[i]/Den-dca*output_parameters[3]*deltapx[i]*deltapy[i]/Den2);
		//dchi2/dty:
		derivatives[3*n_points+i] = +2*values[i]*(deltapx[i]*p1z[i]/Den-dca*output_parameters[2]*deltapx[i]*deltapy[i]/Den2);

		*d_chi2 = 2.0f * m_a * x + m_b;
	}
*/
	__global__ void kernelGradf(const float *d_x, float *d_y, float *d_grad)
	{
		const float &x = *d_x;
		*d_y = m_a * x * x + m_b * x + m_c;
		*d_grad = 2.0f * m_a * x + m_b;
	}

	__global__ void kernel_combined_f_gradf(const float *d_x, float *d_y, float *d_grad)
	{
		const float &x = *d_x;
		*d_y = m_a * x * x + m_b * x + m_c;
		*d_grad = 2.0f * m_a * x + m_b;
	}
}

class gpu_sqtrack_chi2 : public cost_function
{
public:
	gpu_sqtrack_chi2(float a, float b, float c) 
	: cost_function(1) 
	{
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_x0, gpu_sqtrack_chi2_d::m_x0) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_y0, gpu_sqtrack_chi2_d::m_y0) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_tx, gpu_sqtrack_chi2_d::m_tx) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_ty, gpu_sqtrack_chi2_d::m_tx) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_pinv, gpu_sqtrack_chi2_d::m_pinv) );

		CudaSafeCall( cudaMemcpy(m_d_x0, &x0, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_y0, &y0, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_tx, &tx, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_ty, &ty, sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_pinv, &pinv, sizeof(float), cudaMemcpyHostToDevice) );
	}
	
	void f_gradf(const float *d_x, float *d_f, float *d_grad)
	{
		gpu_sqtrack_chi2_d::kernel_combined_f_gradf<<<1, 1>>>(d_x, d_f, d_grad);
		CudaCheckError();
	}
	
private:

	float *m_d_x0; 
	float *m_d_y0;
	float *m_d_tx;
	float *m_d_ty;
	float *m_d_pinv;
};

/*
int main(int argc, char **argv)
{
	// CPU

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

	return 0;
}
*/
