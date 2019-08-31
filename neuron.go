package main

import (
	"math"
)

type signal struct {
	x, y int
}

type neuron struct {
	weightX, weightY float64
	bias             float64
}

func sigmoid(x float64) float64 {
	// f(x) = 1 / (1 + e^(-x))
	return 1.0 / (1 + math.Exp(-x))
}

func (n neuron) feedForward(s signal) float64 {
	fx, fy := float64(s.x), float64(s.y)
	total := sigmoid(fx*n.weightX + fy*n.weightY + n.bias)
	return sigmoid(total)
}
