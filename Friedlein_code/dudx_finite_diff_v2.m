function dudx = dudx_finite_diff_v2(x,u)
%Caluculates the finite difference approximation to the first derivative of
%u with respect to x.
%   This function uses a fourth-order accurate approximation to the first
%   derivative of u at all grid points x. It uses centered finite
%   differences at all non-edge points and one-sided approximations at the
%   edge points.
%
%   In v2, x must be uniformly spaced.
%
%dudx - first derivative of u with respect to x. This is a vector with the
%same dimensions as u.
%
%x - independent variable. This should be a 1D vector with the same length as
%u, and it must be uniformly spaced (ie diff(x)=const.)
%
%u - dependent variable. This is a 1D vector defined such that
%u(i)=u(x=x(i)).

%Get the spacing of the grid points.
h=mean(diff(x));

%Get the centered finite difference approximations at all non edge points.
u_cen1=u(1:end-4);
u_cen2=u(2:end-3);
u_cen3=u(3:end-2);
u_cen4=u(4:end-1);
u_cen5=u(5:end);
dudx_cen=1/(12*h)*(u_cen1-8*u_cen2+8*u_cen4-u_cen5);

%Get the one-sided finite difference approx at the left edge.
dudx_left=1/(12*h)*(-25*u(1)+48*u(2)-36*u(3)+16*u(4)-3*u(5));
%Get the finite difference approx at 1 point inside the left edge.
dudx_left1=1/(12*h)*(-3*u(1)-10*u(2)+18*u(3)-6*u(4)+u(5));

%Get the one-sided approx at the right edge.
dudx_right=1/(12*h)*(3*u(end-4)-16*u(end-3)+36*u(end-2)-48*u(end-1)+25*u(end));
%Get the finite diff approx 1 point inside the right edge.
dudx_right1=1/(12*h)*(-u(end-4)+6*u(end-3)-18*u(end-2)+10*u(end-1)+3*u(end));
dudx=[dudx_left,dudx_left1,dudx_cen,dudx_right1,dudx_right];


end

