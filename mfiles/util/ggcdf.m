function [p] = ggcdf(x,a,b)
if nargin<2
    error('stats:laplcdf:TooFewInputs','Input argument X is undefined.');
end
p = zeros(length(x),1);
pmin = find(x <= 0);
pmax = find(x >  0);
p(pmin) = plica(1/b,(-x(pmin)/a).^b)./(2*gamma(1/b));
p(pmax) = 1- plica(1/b,(x(pmax)/a).^b)./(2*gamma(1/b));
