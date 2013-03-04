function r = ggrnd(a,b,m,n);
%GGRND Random matrices from generalized Gaussian distribution.
%   R = GGRND(A,B) returns a matrix of random numbers chosen   
%   from the generalized Gaussian distribution with parameters A and B.
%   The size of R is the common size of A and B if both are matrices.
%   If either parameter is a scalar, the size of R is the size of the other
%   parameter. Alternatively, R = GAMRND(A,B,M,N) returns an M by N matrix. 
%
%   PDF for generalized Gaussian distribution is defined as:
%   	F(X) = (B/(2*A*gamma(1/B))) * exp(-(X/A)^B)

%   GGRND uses the fact that if V is uniformly distributed on [-A, +A]
%   and Y is gamma(1+1/B,1) distributed then X = V*Y^(1/B) is gg(A,B)
%
%   See also: GAMRND

%   References:
%      [1]  L. Devroye, "Non-Uniform Random Variate Generation", 
%      Springer-Verlag, 1986

if nargin < 2, 
   error('Requires at least two input arguments.'); 
end


if nargin == 2
   [errorcode rows columns] = rndcheck(2,2,a,b);
end

if nargin == 3
   [errorcode rows columns] = rndcheck(3,2,a,b,m);
end

if nargin == 4
   [errorcode rows columns] = rndcheck(4,2,a,b,m,n);
end

if errorcode > 0
   error('Size information is inconsistent.');
end

% Initialize
lth = rows*columns;
a = a(:); b = b(:);

scalara = (length(a) == 1);
if scalara 
   a = a*ones(lth,1);
end

scalarb = (length(b) == 1);
if scalarb 
   b = b*ones(lth,1);
end

% V
v = 2 * a .* (rand(lth,1) - 0.5);

% Y
y = gamrnd(1 + 1 ./ b, 1);

% R
r = v .* y .^ (1 ./ b);

% Return NaN if a or b is not positive.
r(b <= 0 | a <= 0) = NaN;

r = reshape(r,rows,columns);