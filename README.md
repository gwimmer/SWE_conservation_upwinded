# SWE_conservation_upwinded

Repository containing code for paper:
"Energy conserving upwinded compatible finite element schemes
for the rotating shallow water equations"
by Golo Wimmer, Colin Cotter, Werner Bauer
 
 REQUIREMENTS
 The code relies on the firedrake FEM package; to install it see
 https://firedrakeproject.org
 Further, for diagnostic output the netCDF4 package is used.
 
 Note: the code is suitable for parallel runs;  a script for
 running the W2 test case on the general purpose cluster
 CX1 of Imperial College London is given in script.pbs
 (assumes firedrake installed in $HOME, relevant python 
 file and script in $WORK/W2).
 For information on how to install firedrake on CX1 see
 https://github.com/firedrakeproject/firedrake/wiki/CX1
 
 FILES CONTAINED
 The test cases are given by:
 PUSM - Periodic unit square mesh test case
 W2 - Williamson 2 test case
 W5 - Williamson 5 test case
 Galewsky - Galewsky test case
 
 The discretisation setups are given by:
 ec: energy conserving space discretisation
            -D-ad: includes upwinding in D
            -flux: does not include upwinding in D (flux form instead)
 nec: non-energy conserving space discretisation

 All test cases are run with Poisson time integrator and a Picard 
 iteration scheme (as described in paper).

 For each file, the parameter setup (resolution, time step, tmax, 
 Picard iteration nr, etc.) is given as in the paper. For output
 options, see files.
