# PredictiveModel

Develop and evaluate an algorithm to discover the mathematical model of molecular interactions. Authors: Sheldon Zhu, Eudora Fong, Joseph Perez

Inputs 
c0 <- initial concentrations 
v0 <- initial volumes
∇c <- added concentrations
∇v <- added volume
x <- initial spec names

Helper Functions
Generate species names 
Inputs
spec_name <- {n1, n2, …, nn} (Prior species within the chemical model)
Combinatorics (with replacement)
CR(n,r) = C(n+r-1,r) = (n+r-1)! / r!
Generate new models
Transform species names into model configuration
For each new model, use NGLM to optimize betas for each species
fname <- rcalc (function)
p, t, y
ssqold <- 1e50
Mp <- 0 (Marquardt parameter) 
Μ <- 1e-4 (convergence limit)
δ <- 1e-6 (step size for numerical diff)
it <- 0
while it<50 
r0=fname(p,t,y); (call calc of residuals) 
ssq=sum(r0.*r0); 
conv_crit=(ssq_old-ssq)/ssq_old; 
if abs(conv_crit) <= mu (ssq_old=ssq, minimum reached ! )
if mp==0 
break 
if Marquardt par zero, 
stop 
else 
otherwise mp=0; (set to 0 , another iteration) 
r0_old=r0; 
end else
if conv_crit > mu (convergence ! )
mp=mp/3; 
ssq_old=ssq; 
r0_old=r0; 
for i=1:length(p) 
p(i)=(1+delta)*p(i); 
r=feval(fname,p,t,y); 
J(:,i)=(r-r0)/(delta*p(i)); 
p(i)=p(i)/(1+delta); 
end 
elseif conv_crit < -mu  (divergence ! )
if mp==0 
mp=1; (use Marquardt parameter) 
else 
mp=mp*5; 
end 
p=p-delta_p; (and take shifts back )
end 
J_mp=[J;mp*eye(length(p))]; (augment Jacobian matrix) r0_mp=[r0_old;zeros(size(p))]; (augment residual vector) 
delta_p=-J_mp\r0_mp; (calculate parameter shifts)  
Test new models
Simulate pH data with Newton-Raphson given model and betas
Model, beta, c_tot,c, i
ncomp=length(c_tot); % number of components 
nspec=length(beta); % number of species 
c_tot(c_tot==0)=1e-15; % numerical difficulties if c_tot=0 
it=0; 
while it<=99 
it=it+1; 
c_spec =beta.*prod(repmat(c',1,nspec).^Model,1); %species conc c_tot_calc=sum(Model.*repmat(c_spec,ncomp,1),2)'; %comp ctot calc 
d =c_tot-c_tot_calc; % diff actual and calc total conc 
if all(abs(d) <1e-15) % return if all diff small 
return 
end 
for j=1:ncomp % Jacobian (J*) 
for k=j:ncomp 
J_s(j,k)=sum(Model(j,:).*Model(k,:).*c_spec); J_s(k,j)=J_s(j,k); % J_s is symmetric 
end 
end 
delta_c=(d/J_s)*diag(c); % equation (2.43) 
c=c+delta_c; 
while any(c <= 0) % take shift back if conc neg. 
delta_c=0.5*delta_c; 
c=c-delta_c; 
if all(abs(delta_c)<1e-15) 
break 
end 
end 
end
 Diff = Euclidean distance of pH meas real and sim pH meas
d(p,q) = i=1n(qi-pi)2 ,
where p, q are two points in Euclidean n-space,
qi, pi are Euclidean vectors, starting at the initial point,
and n is the domain dimension.

SR calculation
If SR > 1000, iteratively output SR2 else SR2 = 0

Main
Queue = apply generate species name
Top species names = empty list top species names
Iterate through queue
Apply generate new model
Apply test new model
If length of top species names < 5, append to names
Else if the last element of top species names has larger diff than current iteration, replace, sort top species names
Return top 5 elements in top species names

