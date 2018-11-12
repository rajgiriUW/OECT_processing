function eta_calc=eta_v5_erf(ymax,sig_DOS,E0,EF,T,n)
%This function calculates the right-hand side of equation (11) in the
%Dunlap conductance paper (see ...
%/Desktop/Organic electronic papers/AHL Conductance - Dave Dunlap.pdf)
%
%In v1, I use loops for the double integral.
%
%In v3_gift, I use nested trapz calls for the double integral. Also,
%there's a subtle but important difference in the way eta_calc is indexed.
%In this version, eta_calc is always a vector (never 2D), and eta_calc(i)
%is solved for the case when the Fermi level is EF(i) AND ymax is ymax(i).
%So, EF, and ymax are always paired. They have to have the same length.
%This version is meant to be wrapped inside a function that finds the value
%of ymax which satisfies equation 11 for a fixed eta.
%
%In v4, I get rid of the temperature normalization for my energy variables.
%They are now in SI units (joules)
%
%In v5, I use an erf to calculate the value of the inner integral.
%Also, in v5, I get rid of the weird pairwise indexing mentioned in the v3
%notes. That means that eta_calc(i,j) is the calculated eta value when ymax
%is ymax(i) and EF is EF(j). 
%Also, in v5, I demand that sig_DOS, E0, EF, T, and n are all scalar
%values.
%
%eta_calc: value of the right-hand side of equation (11). eta_calc(i,j) is
%the value for ymax(i) and EF(j).
%
%ymax: upper bound of the dimensionless energy difference such that the
%conductance between sites with that energy difference is greater than or
%equal to the critical conductance, gmin. This is a dimensionless vector
%
%sig_DOS: width of the standard deviation of the Gaussian DOS. Scalar
%
%EO: Energy at which the peak of the DOS is located. Scalar, J
%
%EF: Fermi level (chemical potential) of the system. Vector, J.
%
%T: Temperature of the system in K. scalar
%
%n: number of discretization steps in the numerical integrals. scalar

k=1.38e-23; %Boltzmann constant.

%Make sure both EF and ymax are column vectors.
EF=reshape(EF,[],1);
ymax=reshape(ymax,[],1);

%Meshgrid EF and ymax.
[ymax1,EF1]=meshgrid(ymax,EF);

%Find the dimensionality of EF and ymax.
szymax=size(ymax);
ndim_ymax=length(find(szymax>1));%If scalar, ndim=0

szEF=size(EF);
ndim_EF=length(find(szEF>1));%If scalar, ndim=0

%Calculate r as a function of ymax. Equation 7 of the Dunlap document.
r=4*exp(-abs(ymax1))./(2*cosh(ymax1)+2);

%Define the outer variable of integration, y. It ranges from -ymax to ymax.
%Because ymax could be a vector, we use linspace NDim. Each column of the
%resulting array will correspond to a single value of ymax.
y=linspaceNDim(-ymax1,ymax1,n);

%Repmat r so that it will be the same shape as y.
if ndim_ymax==1
    if ndim_EF==1
        r=repmat(r,1,1,n);
    else
        r=repmat(r,n,1)';
    end
else
    if ndim_EF==1
        r=repmat(r,1,n);
    else
        r=repmat(r,1,n)';
    end
end
     

%Calculate xmax for each value of y. Define an intermediate variable z such
%that cosh(z)=xmax.
z=2*exp(-abs(y))./r-cosh(y);
%This is necessary to control floating point errors when y=+/- ymax
switch ndim_EF+ndim_ymax
    case 0
        z(1)=1;%y=-ymax
        z(end)=1;%y=ymax
    case 1
        z(:,1)=1;%y=-ymax
        z(:,end)=1;%y=ymax
    case 2
        z(:,:,1)=1;%y=-ymax 
        z(:,:,end)=1;%y=ymax
end

xmax=acosh(z);

%Make EF the same shape as y.
if ndim_ymax==1
    if ndim_EF==1
        EF1=repmat(EF1,1,1,n);
    else
        EF1=repmat(EF1,n,1)';
    end
else
    if ndim_EF==1
        EF1=repmat(EF1,1,n);
    else
        EF1=repmat(EF1,n,1);
    end
end

%Set the zero point of the Fermi energies to the peak of the DOS.
EF1=EF1-E0;

%When using a gaussian DOS, the inner integral in (11) of the Dunlap paper
%is an erf, but we still need to integrate over y.
integrand=k*T/(2*sig_DOS*sqrt(pi))*...
    exp(-(k*T*y).^2/sig_DOS^2).*...
    (erf((k*T*xmax+EF1)/sig_DOS)-erf((-k*T*xmax+EF1)/sig_DOS));

%We have to be careful about how to deal with different numbers of
%dimensions for y. We always want to integrate over the last non-singleton
%dimension of y.
switch ndim_ymax+ndim_EF
    case 0
        %Make sure to multiply by the right dy spacing for each ymax value.
        dy=y(2)-y(1);
        eta_calc=dy*trapz(integrand);
    case 1
        dy=diff(y,[],2);
        eta_calc=squeeze(dy(:,1)).*trapz(integrand,2);
    case 2
        dy=diff(y,[],3);
        eta_calc=trapz(integrand,3).*squeeze(dy(:,:,1));
end
eta_calc=eta_calc';

%Save variables for debugging.
assignin('base','EF1_v5',EF1)
assignin('base','integrand_v5',integrand)
assignin('base','dy_v5',dy)
assignin('base','z_v5',z)
assignin('base','r_v5',r)


end
