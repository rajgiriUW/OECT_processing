function ymax = solve_eta_ymax_v2(eta,sig_DOS,E0,EF,T,n,approx)
%This function finds the value of ymax that gives the desired eta in
%equation 11 of Dunlap's notes: "AHL conductance". see ...
%/Desktop/Organic electronic papers/AHL Conductance - Dave Dunlap.pdf)
%
%In v2, I don't normalize the energies to kT. They are in J
%
%ymax: Upper bound of the normalized variable describing the energy
%difference between two hopping sites.
%
%eta: fraction of connections with conductance greater than gmin. This is
%the critical conductance. Let's keep this guy as a scalar for now.
%
%sig_DOS: width of the standard deviation of the Gaussian DOS. Scalar in J
%
%EO: Energy at which the peak of the DOS is located. Scalar in J
%
%EF: Fermi level (chemical potential) of the system. Scalar in J.
%
%T: Temperature, in K. Scalar
%
%n: number of discretization steps in the numerical integrals.
%
%approx: flag describing whether to use the approximation or the full
%expression. approx is a boolean scalar. if true, the approximation will be
%used. This input is optional. If you leave it blank, the full expression
%will be used.

k=1.38e-23; %Boltzmann constant in J/K
    function cost=rapper(ym)
        %This function just wraps the eta_v3_gift function defined in
        %...\BEL\Shen_model_2016\ It passes in the parameters sig_DOS, E0,
        %and n from the outer function, and it solves eta for each value of
        %EF requested in the outer function.
        %
        %cost is a column vector where cost(i) is the difference between
        %the eta value we want (input into the outer function) and the eta
        %value we get when ymax=ym. cost(i) is this difference when the
        %Fermi level is EF(i) and ymax is ym(i).
        cost=abs((eta_v5_erf(ym,sig_DOS,E0,EF,T,n)-eta)/eta);
    end

    %Define the approximate expression. This is equation (18) in the Dunlap
    %document.
    function cost=rapper_approx(ym)
        y=linspace(0,ym,n);%Variable of integration
        c=(2*cosh(ym)+2).*exp(ym-abs(y))-2*cosh(y);%Definition of c(y)
        c(y==ym)=2;
        c(y==-ym)=2;
        integrand=log(c/2+sqrt(c.^2/4-1));%Integrand in eqn 18
        rho_EF=1/(sig_DOS*sqrt(2*pi))*exp(-(EF-E0)^2/(2*sig_DOS^2));%DOS at EF
        eta_calc=8*(k*T)^2*rho_EF^2*trapz(y,integrand);
        cost=abs(eta_calc-eta);
    end

%     function cost=rapper_approx(ym)
%         %This function just wraps the approximate function defined in
%         %...\BEL\Shen_model_2016\ - equation 7.
%         %
%         %This approximation has less than .07% error for ymax>=8
%         %
%         %cost is a column vector where cost(i) is the difference between
%         %the eta value we want (input into the outer function) and the eta
%         %value we get when ymax=ym. cost(i) is this difference when the
%         %Fermi level is EF(i) and ymax is ym(i).
%         cost=eta_v4_gift(ym,sig_DOS,E0,EF,T,n)-eta;
%     end

if approx
    fun_hand=@rapper_approx;
else
    fun_hand=@rapper;
end

eta_calc0=0;
ymax0=1;
i=1;
cost=1e9;
while eta_calc0<eta&&i<200&&abs(cost)>1e-1
    eta_calc0=eta_v5_erf(ymax0,sig_DOS,E0,EF,T,n);
    options = optimoptions('fsolve','Display','off');
    ymax0 = ymax0+0.25;
%     if ymax0<8
        [ymax,cost]=fsolve(fun_hand,ymax0,options);
%     else
%         [ymax,cost]=fsolve(@rapper_approx,ymax0,options);
%     end
    i=i+1;
end
% options = optimoptions('fsolve','Display','off');
% ymax0 = 1;
% [ymax,cost]=fsolve(fun_hand,ymax0,options);

if abs(cost)>.05*eta
    error(['ymax not found. EF is ',num2str(EF),...
        ' sig_DOS is ',num2str(sig_DOS),' eta is ',num2str(eta)])
end
end


