#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:43:45 2021

@author: laurent
"""

import numpy as np
from scipy.linalg import svd

import argparse as args

def nonLinearFunction(inputs, project, reproject) :
	
	interm = project @ inputs
	
	outSize = reproject.shape[0]
	nPoints = interm.shape[1]
	
	outputs = np.zeros((outSize, nPoints))
	outputs += reproject[:,0].reshape(outSize,1)*(0.2*interm[0,:]**2 + 0.2*interm[2,:]*np.sin(interm[1,:]/3) + 0.3*np.abs(interm[2,:])**(1/2) + interm[1,:] + 2).reshape(1,nPoints)
	outputs += reproject[:,1].reshape(outSize,1)*(0.3*interm[1,:]**2 + 0.1*interm[0,:]*np.sin(interm[2,:]/4) + 0.1*np.abs(interm[0,:])**(1/2) + interm[0,:] + 1).reshape(1,nPoints)
	outputs += reproject[:,2].reshape(outSize,1)*(0.1*interm[2,:]**2 + 0.3*interm[1,:]*np.sin(interm[0,:]/5) + 0.6*np.abs(interm[1,:])**(1/2) + interm[2,:] + 2).reshape(1,nPoints)
	
	outputs += 0.1*np.cos(inputs/3.)
	
	return outputs

def generateData(nPoints, nTestPoints = None, mesThreshold = 0.8) :
	
	inSize = 9
	outSize = 9
	noisyChannel = 3
	intermSize = 3
	
	if nTestPoints is None :
		nTestPoints = int(nPoints/10)
		if nTestPoints < 10 :
			nTestPoints = 10

	# Genrate a random projection matrix to R3 and reprojection matrix
	rOrient = 1e-1*(2*np.eye(inSize) + 5*np.random.rand(inSize, inSize))
	rProject = svd(np.random.rand(inSize, intermSize), full_matrices=False)[0].T
	rReproject = svd(np.random.rand(outSize, intermSize), full_matrices=False)[0]

	# Inputs are sampled from a zero-mean multivariate normal distribution with a random correlation matrix	
	inputs = rOrient @ np.random.normal(size = (inSize, nPoints))
	testinputs = rOrient @ np.random.normal(size = (inSize, nTestPoints))
	
	# The input is then multiplied by a random projection matrix to R3 before a non-linear function is applied
	# A random set of three orthonormal vectors in Rm is then generated and used to reproject the values in the final output space, constrained on a known hyperplane.
	outputs = nonLinearFunction(inputs, rProject, rReproject)
	testoutputs = nonLinearFunction(testinputs, rProject, rReproject)
	
	# noise is added to the training data, with one channel receiving much more noise than the other channels
	outputs += 0.1*np.random.normal(size = (outSize, nPoints))
	outputs[noisyChannel,:] += 3*np.random.normal(size = (nPoints))
	
    # The measured values for the training set are then selected at random, the other values being set to 0.
	measured = np.random.rand(outSize, nPoints) > mesThreshold
	outputs[np.logical_not(measured)] = 0
	
	return inputs, outputs, testinputs, testoutputs, measured, rReproject

if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Generate a toy dataset.')
	parser.add_argument("outputfile", help="The filed where to store the toy dataset")
	parser.add_argument("--size", "-n", type=int, help="Number of features/labels pairs to generate")
	
	args = parser.parse_args()
	
	i, o, ti, to, m, reproj = generateData(args.size)
	
	np.savez(args.outputfile, inputs = i, ouputs = o, testinputs = ti, testouputs = to, measured = m, reproj = reproj)