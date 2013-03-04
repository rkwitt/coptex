function y = ggpdf(x, a, b)
% GGPDF	Generalized Gaussian probability density function
%	Y = GGPDF(X, A, B) returns the generalized Gaussian density function
%	with parameters A and B, at the values in X.
%
%	Y = (B/(2*A*gamma(1/B))) * exp(-(X/A)^B)

%   Return NaN if the arguments are outside their respective limits.
if (a <= 0 | b <= 0)
    tmp = NaN;
    y = tmp(ones(size(x)));    
else
    y = log(b) - log(2) - log(a) - gammaln(1/b) - (abs(x).^b) ./ (a^b);
    y = exp(y);
end