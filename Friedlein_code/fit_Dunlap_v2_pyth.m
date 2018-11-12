function z=fit_Dunlap_v2_pyth(...
    ID,VG,VD,sig_cond0,VG0,E00,sig_DOS0,eta0,T,n,W,L,h)
%This function fits a transfer curve to the Dunlap model using Matlab's
%nonlinear fitter nlinfit.
%
%v2 refers to the Second version of the Dunlap model, not v2 of the fitting
%routine.
%
%v2_pyth is the python compatible version. The way I make it compatible is
%to output a single list of values rather than have 4 separate outputs.
%
%Output: z=[E0,sig_DOS,eta,ci(1),ci(2)]; See below for each element in z
%
%EO: Energy at which the peak of the DOS is located. Scalar in J
%
%%eta: percolation fraction. Percolative transport will occur when a
%fraction eta of all connections have conductance greater than the critical
%conductance. Unitless scalar
%
%sig_DOS: width of the standard deviation of the Gaussian DOS. Scalar in J
%
%ci: Confidence intervals for the parameter estimates. This is estimated
%using the covariance estimate from nlinfit. Probably not terribly
%accurate.
%
%   The matrix ci is defined as follows
%   
%  ci = [E0_lb          E0          E0_ub;
%        sig_DOS_lb     sig_DOS     sig_DOS_ub;
%        eta_lb         eta         eta_ub];
%
%where x_lb is the lower bound of the 95% confidence interval on the
%estimate of parameter x, and x_ub is the upper bound of the 95% confidence
%interval on the estimate of parameter x.
%
%ID: Experimental drain current. Vector in A
%
%VG: Experimental gate voltage. ID(i) was measured at VG(i). Vector in V
%
%VD: Experimental drain voltage. Scalar in V
%
%sig_cond0: Experimental value of the conductivity at VG = VG0. Scalar in
%S/m
%
%VG0: Gate voltage at which sig0 is specified. Scalar in V
%
%E00: Initial guess for E0. 3-vector in J. E00(1) is the lower bound on E0,
%E00(2) is the initial guess for E0, and E00(3) is the upper bound.
%
%sig_DOS0: Initial guess for sig_DOS. 3 vector in J. sig_DOS0(1) is the
%lower bound on sig_DOS, sigDOS0(2) is the initial guess for sig_DOS, and
%sig_DOS0(3) is the upper bound.
%
%eta0: Initial guess for eta. Unitless 3-vector. eta0(1) is the lower bound
%on eta, eta0(2) is the initial guess for eta, and eta0(3) is the upper
%bound.
%
%T: Experimental temperature. Scalar in K
%
%n: Number of discretization steps for numerical calculations.
%
%W: Channel width. Scalar in m.
%
%L: Channel length. Scalar in m.
%
%h: Channel width. Scalar in m.

%Constants
q=1.602e-19; %Elementary charge in C

%Calculate the experimental transconductance.
gm_exp=dudx_finite_diff_v2(VG',ID')';%In S

%Normalize the transconductance to it's median value.
med_gm=median(gm_exp);
gm_exp=gm_exp/med_gm;

%Define the lower bound and upper bound on the fit parameters.
lb=[E00(1),sig_DOS0(1),eta0(1)];
ub=[E00(3),sig_DOS0(3),eta0(3)];

%Normalize to the intial guesses.
lb=lb./abs([E00(2),sig_DOS0(2),eta0(2)]);
ub=ub./abs([E00(2),sig_DOS0(2),eta0(2)]);

%Give the initial guess a fancy name.
beta0=[E00(2),sig_DOS0(2),eta0(2)];

%Normalize to the intial guess (this will make everything either 1 or -1).
beta0=beta0./abs([E00(2),sig_DOS0(2),eta0(2)]);

%Define some options.
options = optimoptions('lsqcurvefit','MaxFunctionEvaluations',1000,...
    'MaxIterations',1000,'StepTolerance',1e-12,'FunctionTolerance',1e-8);

%Do the fit with lsqcurvefit. This lets me use parameter bounds.
[beta,~,residual,~,~,~,jacobian] = lsqcurvefit(...
    @transconductance,beta0,VG,gm_exp,lb,ub,options);

%Get the confidence intervals. These are approximated via the Jacobian of
%the problem.
ci = nlparci(beta,residual,'jacobian',jacobian);

%Denormalize the fit parameters.
beta=beta.*abs([E00(2),sig_DOS0(2),eta0(2)]);
ci=[ci(1,:)*abs(E00(2));
    ci(2,:)*abs(sig_DOS0(2));
    ci(3,:)*abs(eta0(2))];

%Redefine the confidence intervals to have the form specified in the doc
%above.
ci = [ci(:,1),beta',ci(:,2)];

%Rename the elements of beta according to the Dunlap model.
E0=beta(1);
sig_DOS=beta(2);
eta=beta(3);

%Wrap the outputs into a single variable.
z=[E0,sig_DOS,eta,ci(1),ci(2)];

    function gm = transconductance(params,VG1)
        %This function calculates the transconductance of an OECT.
        %
        %gm: Transconductance. Vector in S. gm(i) corresponds to VG1(i).
        %
        %params: Parameter vector.
        %
        %   params(1) = E0/E00(2) (see above)
        %   params(2) = sig_DOS/sig_DOS0(2) (see above)
        %   params(3) = eta/eta0(2) (see above)
        %
        %VG1: Experimental gate voltages. Vector in V. VG1(i) corresponds
        %to gm(i).
        
        %First, some normalization stuff.
        
        %Calculate the conductivity at VG0, normalized to sigma_max, not sig_cond0.
        %Then, the conductivity at all voltages is given by
        %
        %        sig_cond = sig_cond0/r0*2*exp(-|ymax|)/(cosh(ymax)+1)
        %
        %where r0 is the conductivity at VG0 divided by sigma_max.
        
        %Denormalize the fit parameters.
        eta1=params(3)*abs(eta0(2));
        sig_DOS1=params(2)*abs(sig_DOS0(2));
        E01=params(1)*abs(E00(2));
        EF10=q*VG0;
        
        r0=conductivity_Dunlap_v3_scalar(...
            eta1,sig_DOS1,E01,EF10,T,n);
        
        %Calculate the conductivity (normalized to sigma_max) for all VG.
        sig_cond=conductivity_Dunlap_v3_scalar(...
            eta1,sig_DOS1,E01,q*VG1,T,n);%q*VG is the EF at VG.
        
        sig_cond=sig_cond/r0*sig_cond0; %Now it's in S/m
        
        %Now convert this to a transconductance for our OECT geometry.
        %First get the drain current.
        ID_mod=sig_cond*VD*W*h/L;
        %Now take the derivative to get the transconductance.
        gm=dudx_finite_diff_v2(VG1',ID_mod')'; 
        %Normalize to the median experimental transconductanc.
        gm=gm/med_gm;
       
    end

end


