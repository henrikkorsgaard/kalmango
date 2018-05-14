package Kalmango

type JSONfile struct {
	NoiseConstant         []int     `json:"noise_constant"`
	NoiseConstantKalman   []float64 `json:"noise_constant_kalman"`
	NoiseConstantGoKalman []float64 `json:"noise_constant_gokalman"`
}

type Kalman struct {
	R float64
	Q float64
	A float64
	B float64
	C float64

	cov float64
	x   float64
}

func NewKalmanFilter(r float64, q float64, a float64, b float64, c float64) (kal Kalman) {
	kal = Kalman{R: r, Q: q, A: a, B: b, C: c}
	return kal
}

func (k *Kalman) Filter(z float64, u float64) float64 {
	if k.cov == 0 {
		k.x = (1 / k.C) * z
		k.cov = (1 / k.C) * k.Q * (1 / k.C)
	} else {
		predX := k.predict(u)
		predCov := k.uncertainty()

		K := predCov * k.C * (1 / ((k.C * predCov * k.C) + k.Q))

		k.x = predX + K*(z-(k.C*predX))
		k.cov = predCov - (K * k.C * predCov)
	}

	return k.x
}

func (k *Kalman) predict(u float64) float64 {
	return (k.A * k.x) + (k.B * u)
}

func (k *Kalman) uncertainty() float64 {
	return ((k.A * k.cov) * k.A) + k.R
}
