# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:40:23 2022

@author: sheld
"""

import numpy as np
import itertools
import random
from NewtonRaphson import NewtonRaphson
from nglm import nglm
from Rcalc_pH import Rcalc_pH

def predictModel(pH_meas_real, init_spec_name, c_0,c_added, v_0, v_inc, max_elements):
    """
    Predict the chemical model given a list of pH measurements and initial species

    Parameters
    ----------
    pH_meas_real : TYPE list of floats
        DESCRIPTION. List of pH measurements
    init_spec_name : TYPE list
        DESCRIPTION. List of species that the user started with
    c_0 : TYPE list of ints
        DESCRIPTION. List of initial concentrations for each species
    c_added : TYPE list of ints
        DESCRIPTION. List of concentrations added for each species
    v_0 : TYPE int
        DESCRIPTION. Initial volume
    v_inc : TYPE int 
        DESCRIPTION. Volume increments between solutions
    max_elements: TYPE int
        DESCRIPTION. Number of elements that the last species in the model may have.
            Determines stopping point for model generation algorithm.

    Returns
    -------
    TYPE List of lists (2-d list)
        DESCRIPTION. Species name of the top 5 models

    """
    pH_meas_real = np.array(pH_meas_real)
    c_0 = np.array(c_0)
    c_added=np.array(c_added)
    ncomp=np.size(c_0)
    nvol=np.size(pH_meas_real)
    if len(c_0)!=len(c_added):
        return 'Initial Concentration needs to have same dimension as Added Concentration'
    v_added = np.arange(0,v_inc*nvol,v_inc)
    v_tot   = v_0+v_added
    C_tot = np.zeros((nvol, ncomp), dtype=float)
    
    initial_betas = [0]*ncomp+[1]
    best_sigk=np.array([1])
    for j in range(ncomp):
        C_tot[:,j]= (v_0*c_0[0,j]+v_added*c_added[0,j])/v_tot
        
    def generateSpeciesNames(spec_name):
        """
        Generates a new list of species that we want to check and add it to the queue

        Parameters
        ----------
        spec_name : TYPE list of strings
            DESCRIPTION. Prior species within the chemical model
        nelements : TYPE int
            DESCRIPTION. Number of elements belonging to the species we will in the chemical model

        Returns
        -------
        list
            DESCRIPTION. New list of species to check for fit

        """
        comb_list = list(itertools.combinations_with_replacement(spec_name[0:ncomp], len(spec_name)))
        comb_list = [''.join(sorted(s)) for s in comb_list]
        
        #Tentatively creating dictionary to generate unique random betas
        #NEXT STEP: Calculating betas
        #betas will depend on species name as well as species order
        #ex. AH vs HA, H2A vs AH2
        # for comb in comb_list:
        #     if comb not in beta_spec_list:
        #         beta = np.random.randint(18, 30)
        #         beta_spec_list[comb] = beta
        return [spec_name + [s] for s in comb_list]
    
    def generateChemModel(speciesName, initial_betas, best_sigk):
        """
        Generates 2-d array model of elements given species names

        Parameters
        ----------
        speciesName : TYPE 1-d list
            DESCRIPTION. List of species names

        Returns
        -------
        model : TYPE 2-d array of ints
            DESCRIPTION. Chemical model of elements
        betas : TYPE 2-d array of floats
            DESCRIPTION. Betas values for each species (column in the model)

        """
        model = []
        for i in range(len(speciesName)):
            vec=[]
            for spec in init_spec_name:
                vec.append(speciesName[i].count(spec))
            model.append(vec)
        model = np.array(model).T
        #generate betas
        # betas = [beta_spec_list[comb] for comb in speciesName]
        # betas = 10**np.array([betas], dtype='float64')
        
        #generating betas using nglm/optimization algorithm
        #initilize starting betas as starting points
        log_beta_list = initial_betas+[initial_betas[-1]]*(len(speciesName)-len(initial_betas))
        log_beta = np.array([log_beta_list], dtype='float64')
        absorbing = [1 if x=='H' else 0 for x in init_spec_name ]+ [0]*(len(speciesName)-ncomp)
        par_loose = list(range(ncomp, len(speciesName)))
        print(len(speciesName), speciesName, log_beta)
        log_beta,ssq,C,pH_cal,Curv = nglm(Rcalc_pH,log_beta,C_tot,pH_meas_real,model,par_loose,absorbing)	      
        betas = 10**log_beta
        sig_r = np.sqrt(ssq/(np.prod(np.shape(pH_meas_real)) - len(log_beta)));
        try:
            sig_k = sig_r * np.sqrt(np.diag(np.linalg.inv(Curv)));	
        except:
            print('Error getting sig_k: ', log_beta)
            return model, betas, initial_betas, best_sigk
        # print('Printing sig_k: ', sig_k, model)
        # print(best_sigk)
        if np.linalg.norm(sig_k, 2)<np.linalg.norm(best_sigk, 2):
            
            initial_betas=log_beta.tolist()[0]
            print('Initial betas changed to:', initial_betas)
            return model, betas, initial_betas, sig_k
        return model, betas, initial_betas, best_sigk
    
    
    def testModel(chem_model, beta, C_tot):
        """
        Replicates the Data_pH.m matlab file to test if the model's pH_meas is close to pH_meas_real

        Parameters
        ----------
        chem_model : TYPE 2-d array of ints
            DESCRIPTION. Chemical model of elements
        betas : TYPE 2-d array of floats
            DESCRIPTION. Betas values for each species (column in the model)
        C_tot : TYPE 2-d array
            DESCRIPTION. Total concentrations for each solution

        Returns
        -------
        diff : TYPE np.double
            DESCRIPTION. Difference between measured pH from algorithm and real measured pH, taken as inner product

        """
        C=np.zeros((nvol, np.size(beta)))
        c_comp_guess = np.array([[1,1]])*1e-10
        for i in range(nvol):
            #print(C_tot[i, :])
            try:
                C[i, :]=NewtonRaphson(chem_model, beta, C_tot[i, :], c_comp_guess, i)
            except RuntimeError as pH_gen:
                raise RuntimeError from pH_gen
                return 
            c_comp_guess=np.array([C[i,:ncomp]]);
        pH_calc=-1*np.log10(C[:, 1])
        sig_R = 0.002
        pH_meas=pH_calc+sig_R*np.random.random(pH_calc.shape)


        ##### SR #####
        #pH_meas2=[[i] for i in pH_meas]
        pH_meas_real2=[[i] for i in pH_meas_real]
        pH_calc2=[[i] for i in pH_calc]

      
        #Beta=np.linalg.inv(np.transpose(pH_calc2)*pH_calc2)*np.transpose(pH_calc2)*pH_meas_real # Formula referenced in paper
        Beta=np.dot(np.array(pH_calc2),np.array(pH_meas_real2).T)#/np.sqrt(((np.array(pH_calc2)-np.mean(pH_calc))**2)+((np.array(pH_meas_real2)-np.
        wTP=Beta/np.linalg.norm(Beta);
        #print(wTP)
        
        
        tTP=pH_meas*wTP;
        #print(tTP)
        pTP=np.divide((np.transpose(tTP)*pH_meas),(np.transpose(tTP)*tTP));
        #print(pTP)
        #print(tTP*np.transpose(pTP))
        E_TP=pH_meas-(tTP*np.transpose(pTP));

        #Normal SR
        SR=(np.var(tTP*pTP))/(np.var(E_TP))




        diff = np.linalg.norm(pH_meas_real-pH_meas, ord=2)

        if SR > 1000:
          SR2=[]
          for i in range(len(C.T)):
            C2=[[x] for x in C.T[i]]
            Beta=np.dot(np.array(np.log10(C2)),np.array(pH_meas_real2).T)
            wTP=Beta/np.linalg.norm(Beta);
            tTP=pH_meas*wTP;
            pTP=np.divide((np.transpose(tTP)*pH_meas),(np.transpose(tTP)*tTP));
            E_TP=pH_meas-(tTP*np.transpose(pTP));
            SR_temp=(np.var(tTP*pTP))/(np.var(E_TP))
            SR2.append(SR_temp)
          return diff,SR,SR2,pH_meas
        SR2=0
        return diff,SR,SR2,pH_meas
    
    np.random.seed(4)
    queue = generateSpeciesNames(init_spec_name)

    
    #List for top 5 predicted species names and their diffs
    topSpeciesNames = []
    
    while len(queue)!=0: 
        
        np.random.seed(4)
        speciesName=queue.pop(0)
        if len(speciesName)-ncomp+1>max_elements:
            continue
        queue.extend(generateSpeciesNames(speciesName))
        
        try:
            chem_model, beta, initial_betas, best_sigk=generateChemModel(speciesName, initial_betas, best_sigk)
            diff,SR,SR2,pH_meas_model=testModel(chem_model, beta, C_tot)
        except RuntimeError:
            continue
        try:
            # print('Beta: ', beta)
            print('Initial betas: ', initial_betas)
            log_beta = np.log10(beta)
            print(speciesName, log_beta)
        except:
            print('A beta equals 0!:', beta)
        
        if len(topSpeciesNames)<5 :
            topSpeciesNames.append((speciesName, diff,SR,SR2,pH_meas_model))
            topSpeciesNames.sort(key = lambda x: x[1])
        elif topSpeciesNames[4][1]>diff:
            topSpeciesNames.pop()
            topSpeciesNames.append((speciesName, diff,SR,SR2,pH_meas_model))
            topSpeciesNames.sort(key = lambda x: x[1])
    print(topSpeciesNames)
    #Use regex to generate valid species name
    #Ask Fereshteh how to format species name strings?
    return [[x[0],x[4]] for x in topSpeciesNames]

