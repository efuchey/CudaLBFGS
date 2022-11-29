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
	cpu_sqtrack_chi2(size_t n_points, float* h_drift, float* h_res, 
			float* h_p1x, float* h_p1y, float* h_p1z, 
			float* h_deltapx, float* h_deltapy, float* h_deltapz) 
	: cpu_cost_function(4),
	m_n_points (n_points),
	m_h_drift (h_drift),
	m_h_res (h_res),
	m_h_p1x (h_p1x),
	m_h_p1y (h_p1y),
	m_h_p1z (h_p1z),
	m_h_deltapx (h_deltapx),
	m_h_deltapy (h_deltapy),
	m_h_deltapz (h_deltapz){}
	
	void cpu_f_gradf(const floatdouble* h_x, floatdouble *h_chi2, floatdouble *h_grad)
	{
		float m_x0 = h_x[0];
		float m_y0 = h_x[1];
		float m_tx = h_x[2];
		float m_ty = h_x[3];
		
		printf("x0 %1.6f, y0 %1.6f, tx %1.6f, ty %1.6f \n", m_x0, m_y0, m_tx, m_ty);
		
		for (size_t n = 0; n<m_numDimensions; n++)
		{
			h_grad[n] = 0.0;
		}
		*h_chi2 = 0.0;
		for(size_t i = 0; i<m_n_points; i++){
			const double Den = sqrt( (m_ty*m_h_deltapz[i] - m_h_deltapy[i])*(m_ty*m_h_deltapz[i] - m_h_deltapy[i]) +
			(m_h_deltapx[i] - m_tx*m_h_deltapz[i])*(m_h_deltapx[i] - m_tx*m_h_deltapz[i]) +
			(m_tx*m_h_deltapy[i] - m_ty*m_h_deltapx[i])*(m_tx*m_h_deltapy[i] - m_ty*m_h_deltapx[i]) );
			
			const double dca = ( (m_ty*m_h_deltapz[i] - m_h_deltapy[i]) * ( m_h_p1x[i] - m_x0 ) +
			(m_h_deltapx[i] - m_tx*m_h_deltapz[i]) * ( m_h_p1y[i] - m_y0 ) +
			 m_h_p1z[i] * ( m_tx * m_h_deltapy[i] - m_ty * m_h_deltapx[i] )  ) / Den;
			
			*h_chi2+= ( m_h_drift[i] - dca )*( m_h_drift[i] - dca ) / m_h_res[i] / m_h_res[i];
			
			h_grad[0]+= 1.e-6 * 2.0f * ( m_h_drift[i] - dca ) * (m_ty*m_h_deltapz[i] - m_h_deltapy[i]) / Den / m_h_res[i] / m_h_res[i];
			
			h_grad[1]+= 1.e-6 * 2.0f * ( m_h_drift[i] - dca ) * (m_h_deltapx[i] - m_tx*m_h_deltapz[i]) / Den / m_h_res[i] / m_h_res[i];
			
			h_grad[2]+= 1.e-6 * -2.0f * ( m_h_drift[i] - dca ) * ( (m_h_p1z[i]*m_h_deltapy[i] - m_h_deltapz[i]*(m_h_p1y[i] - m_y0))
									- dca * ( ( m_h_deltapy[i] * m_h_deltapy[i] + m_h_deltapz[i] * m_h_deltapz[i] ) * m_tx - m_h_deltapx[i] * m_h_deltapz[i] - m_h_deltapx[i] * m_h_deltapy[i] * m_ty )/Den ) / Den / m_h_res[i] / m_h_res[i];
			
										
			h_grad[3]+= 1.e-6 * -2.0f * ( m_h_drift[i] - dca ) * ( ( m_h_deltapz[i]*(m_h_p1x[i] - m_x0) - m_h_p1z[i]*m_h_deltapx[i] )
									- dca * ( ( m_h_deltapx[i] * m_h_deltapx[i] + m_h_deltapz[i] * m_h_deltapz[i] ) * m_ty - m_h_deltapy[i] * m_h_deltapz[i] - m_h_deltapx[i] * m_h_deltapy[i] * m_tx )/Den ) / Den / m_h_res[i] / m_h_res[i];

			//printf("%1.6f, %1.6f, %1.6f, %1.6f, (%1.6f - %1.6f),  (%1.6f - %1.6f) \n", dca, Den, (m_ty*m_h_deltapz[i] - m_h_deltapy[i]), (m_h_deltapx[i] - m_tx*m_h_deltapz[i]), (m_h_p1z[i]*m_h_deltapy[i] - m_h_deltapz[i]*(m_h_p1y[i] - m_y0)), dca * ( ( m_h_deltapy[i] * m_h_deltapy[i] + m_h_deltapz[i] * m_h_deltapz[i] ) * m_tx - m_h_deltapx[i] * m_h_deltapz[i] - m_h_deltapx[i] * m_h_deltapy[i] * m_ty )/Den, ( m_h_deltapz[i]*(m_h_p1x[i] - m_x0) - m_h_p1z[i]*m_h_deltapx[i] ), dca * ( ( m_h_deltapx[i] * m_h_deltapx[i] + m_h_deltapz[i] * m_h_deltapz[i] ) * m_ty - m_h_deltapy[i] * m_h_deltapz[i] - m_h_deltapx[i] * m_h_deltapy[i] * m_tx )/Den );
			
		}
		
		printf("chi2 %1.6f; dchi2/dx0 %1.6f, dchi2/dy0 %1.6f, dchi2/dtx %1.6f, dchi2/dty %1.6f \n", *h_chi2, h_grad[0], h_grad[1], h_grad[2], h_grad[3]);
	}

private:

	size_t m_n_points;
	float *m_h_drift;
	float *m_h_res;
	float *m_h_p1x;
	float *m_h_p1y;
	float *m_h_p1z;
	float *m_h_deltapx;
	float *m_h_deltapy;
	float *m_h_deltapz;
};

namespace gpu_sqtrack_chi2_d
{
//	__device__ size_t m_n_points;
//	__device__ float* m_drift;
//	__device__ float* m_res;
//	__device__ float* m_p1x;
//	__device__ float* m_p1y;
//	__device__ float* m_p1z;
//	__device__ float* m_deltapx;
//	__device__ float* m_deltapy;
//	__device__ float* m_deltapz;

	__global__ void kernel_combined_f_gradf(const size_t n_points, const float* d_drift, const float* d_res, 
		const float* d_p1x, const float* d_p1y, const float* d_p1z, 
		const float* d_deltapx, const float* d_deltapy, const float* d_deltapz, 
		const float* d_x, float& d_chi2, float* d_grad)
	{
		d_chi2 = 0.0f;
		d_grad[0] = 0.0f;
		d_grad[1] = 0.0f;
		d_grad[2] = 0.0f;
		d_grad[3] = 0.0f;
		float m_x0 = d_x[0];
		float m_y0 = d_x[1];
		float m_tx = d_x[2];
		float m_ty = d_x[3];
		//printf("%d %1.6f %1.6f %1.6f %1.6f \n", n_points, d_x[0], d_x[1], d_x[2], d_x[3]);
		for(size_t i = 0; i<6; i++){
			//printf("%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f %1.6f %1.6f \n", d_drift[i], d_res[i], d_p1x[i], d_p1y[i], d_p1z[i], d_deltapx[i], d_deltapy[i], d_deltapz[i]);
			const float Den = sqrt( (m_ty*d_deltapz[i] - d_deltapy[i])*(m_ty*d_deltapz[i] - d_deltapy[i]) +
					(d_deltapx[i] - m_tx*d_deltapz[i])*(d_deltapx[i] - m_tx*d_deltapz[i]) +
					(m_tx*d_deltapy[i] - m_ty*d_deltapx[i])*(m_tx*d_deltapy[i] - m_ty*d_deltapx[i]) );
			
			const float dca = ( (m_ty*d_deltapz[i] - d_deltapy[i]) * ( d_p1x[i] - m_x0 ) +
					(d_deltapx[i] - m_tx*d_deltapz[i]) * ( d_p1y[i] - m_y0 ) +
					d_p1z[i] * ( m_tx * d_deltapy[i] - m_ty * d_deltapx[i] )  ) / Den;
			//const float Den = sqrtf( d_deltapy[i] * d_deltapy[i] * (1.0f+d_x[2]*d_x[2]) + d_deltapx[i] * d_deltapx[i] * (1.0f+d_x[3]*d_x[3]) - 2.0f * d_deltapx[i] * d_deltapy[i] * d_x[2] * d_x[3] );
			//const float dca = (-d_deltapy[i]*(d_p1x[i] - d_x[0]) + d_deltapx[i]*(d_p1y[i] - d_x[1]) + d_p1z[i]*(d_x[2]*d_deltapy[i] - d_x[3]*d_deltapx[i]) ) / Den;
			
			//const float value = ( d_drift[i] - dca ) / d_res[i];
			d_chi2+= ( d_drift[i] - dca )*( d_drift[i] - dca ) / d_res[i] / d_res[i];
			//printf("den = %1.6f dca = %1.6f chi2+ = %1.6f\n", Den, dca, ( d_drift[i] - dca )*( d_drift[i] - dca ) / d_res[i] / d_res[i]);
			
			d_grad[0]+= 2.0f * ( d_drift[i] - dca ) * (m_ty*d_deltapz[i] - d_deltapy[i]) / Den / d_res[i] / d_res[i];
			
			d_grad[1]+= 2.0f * ( d_drift[i] - dca ) * (d_deltapx[i] - m_tx*d_deltapz[i]) / Den / d_res[i] / d_res[i];
			
			d_grad[2]+= -2.0f * ( d_drift[i] - dca ) * ( (d_p1z[i]*d_deltapy[i] - d_deltapz[i]*(d_p1y[i] - m_y0))
									- dca * ( ( d_deltapy[i] * d_deltapy[i] + d_deltapz[i] * d_deltapz[i] ) * m_tx - d_deltapx[i] * d_deltapz[i] - d_deltapx[i] * d_deltapy[i] * m_ty )/Den ) / Den / d_res[i] / d_res[i];
			
			d_grad[3]+= -2.0f * ( d_drift[i] - dca ) * ( ( d_deltapz[i]*(d_p1x[i] - m_x0) - d_p1z[i]*d_deltapx[i] )
									- dca * ( ( d_deltapx[i] * d_deltapx[i] + d_deltapz[i] * d_deltapz[i] ) * m_ty - d_deltapy[i] * d_deltapz[i] - d_deltapx[i] * d_deltapy[i] * m_tx )/Den ) / Den / d_res[i] / d_res[i];
			
			
		}
		
		//printf("chi2 = %1.6f\n", d_chi2);
		//d_chi2 = d_chi2/0.;
	}
}

class gpu_sqtrack_chi2 : public cost_function
{
public:
	gpu_sqtrack_chi2(const size_t n_points, const float* d_drift, const float* d_res, 
			const float* d_p1x, const float* d_p1y, const float* d_p1z, 
			const float* d_deltapx, const float* d_deltapy, const float* d_deltapz) 
	: cost_function(4) 
	{
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_n_points, gpu_sqtrack_chi2_d::m_n_points) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_drift, gpu_sqtrack_chi2_d::m_drift) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_res, gpu_sqtrack_chi2_d::m_res) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_p1x, gpu_sqtrack_chi2_d::m_p1x) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_p1y, gpu_sqtrack_chi2_d::m_p1y) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_p1z, gpu_sqtrack_chi2_d::m_p1z) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_deltapx, gpu_sqtrack_chi2_d::m_deltapx) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_deltapy, gpu_sqtrack_chi2_d::m_deltapy) );
		//CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_deltapz, gpu_sqtrack_chi2_d::m_deltapz) );

		//CudaSafeCall( cudaMalloc((void**)&m_n_points, sizeof(size_t)) );
		CudaSafeCall( cudaMalloc(&m_d_drift, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_res, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_p1x, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_p1y, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_p1z, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_deltapx, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_deltapy, n_points * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_deltapz, n_points * sizeof(float)) );
		
		//printf("%lu %lu \n", n_points, m_n_points);
		CudaSafeCall( cudaMemcpy(&m_n_points, &n_points, sizeof(size_t), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_drift, d_drift, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_res, d_res, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_p1x, d_p1x, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_p1y, d_p1y, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_p1z, d_p1z, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_deltapx, d_deltapx, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_deltapy, d_deltapy, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_deltapz, d_deltapz, n_points * sizeof(float), cudaMemcpyHostToDevice) );
		//printf("%lu %lu \n", n_points, m_n_points);
	}
	
	//h_x is what we want to minimize!
	void f_gradf(const float* d_x, float *d_chi2, float *d_gradf)
	{
		
		gpu_sqtrack_chi2_d::kernel_combined_f_gradf<<<1, 1>>>(m_n_points, m_d_drift, m_d_res, 
									m_d_p1x, m_d_p1y, m_d_p1z, 
									m_d_deltapx, m_d_deltapy, m_d_deltapz,
			 						d_x, *d_chi2, d_gradf);
		CudaCheckError();
	}
		
private:
	
	size_t m_n_points;
	float *m_d_drift;
	float *m_d_res;
	float *m_d_p1x;
	float *m_d_p1y;
	float *m_d_p1z;
	float *m_d_deltapx;
	float *m_d_deltapy;
	float *m_d_deltapz;
};

int main(int argc, char **argv)
{

	size_t maxIter = 500;
	float gradientEps = 1e-12f;

	size_t n_points = 6;
	
	float drift[6] = {0.000000,0.000000,0.000000,0.000000,0.000000,0.000000};
	float res[6] = {0.583412447, 0.583412447, 0.6013103054, 0.6013103054, 0.583412447, 0.583412447};
	float p1x[6] = {-147.8155163, -148.8560617, -106.0241719, -107.0382169, -62.10788208, -63.16341238};
	float p1y[6] = {-132.4463745, -132.4429597, -132.4103432, -132.4310822, -133.184595, -133.1777203};
	float p1z[6] = {1315.288245, 1322.270643, 1340.04129, 1347.021991, 1365.497459, 1372.485019};
	float deltapx[6] = {66.6864102, 66.68640862, -0.7181965721, -0.7181965721, -67.33012658, -67.33012658};
	float deltapy[6] = {263.9972253, 263.9972257, 264.1581506, 264.1581506, 263.8331532, 263.8331532};
	float deltapz[6] = {-0.04549869907, -0.04549869907, 0.6791544575, 0.6791544575, -0.5793643679, -0.5793643679};

		
	//gpu_sqtrack_chi2 p1(n_points, drift, res, p1x, p1y, p1z, deltapx, deltapy, deltapz);
	cpu_sqtrack_chi2 p1(n_points, drift, res, p1x, p1y, p1z, deltapx, deltapy, deltapz);

/*
	floatdouble x[4] = {0.01f, 0.01f, 0.001f, 0.001f};
	
	floatdouble chi2_0 = 0.0f;
	floatdouble gradf_0[4] = {0.0f, 0.0f, 0.0f, 0.0f};

	p1.cpu_f_gradf(x, &chi2_0, gradf_0);
	
	floatdouble gradf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	
	floatdouble dx[4] = {0.001f, 0.1f, 0.0001f, 0.0001f};
	
	for(int i = 0; i<4; i++){
		floatdouble dchi2 = 0;
		x[i]+= dx[i];
		p1.cpu_f_gradf(x, &dchi2, gradf);
		printf("%d %1.6f %1.6f %1.6f %1.6f\n", i, (dchi2-chi2_0)/dx[i], gradf_0[i], (dchi2-chi2_0)/dx[i]/gradf_0[i], gradf[i]);
		x[i]-= dx[i];
	}
*/
	lbfgs minimizer(p1);
	minimizer.setMaxIterations(maxIter);
	minimizer.setGradientEpsilon(gradientEps);
	
	float x[4] = {0.0, 0.0, 0.0, 0.0};
	//float x[4] = {75.0, -25.0, -0.1, 0.05};
	//float x[4] = {94.919998, -50.00, -0.14992, 0.074924};

	lbfgs::status stat = minimizer.minimize_with_host_x(x);

/*
	float *d_x;
	CudaSafeCall( cudaMalloc(&d_x,   4 * sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_x, x, 4 * sizeof(float), cudaMemcpyHostToDevice) );
	
	
	//lbfgs::status stat = minimizer.minimize(d_x);
	
	cout << lbfgs::statusToString(stat).c_str() << endl;

	CudaSafeCall( cudaMemcpy(x, d_x, 4 * sizeof(float), cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaFree(d_x) );
*/
	cout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << endl;
	
	return 0;
}



