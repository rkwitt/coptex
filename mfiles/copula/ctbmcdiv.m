function KL = ctbmcdiv(A,B,varargin)
    p = inputParser;
    p.addRequired('A',@isstruct);
    p.addRequired('B',@isstruct);
    p.addParamValue('margin','Weibull',@(x)any(strcmpi(x,{'Weibull','Gamma','Rayleigh','GGD','Cauchy'})));
    p.addParamValue('copula','Gaussian',@(x)any(strcmpi(x,{'Gaussian','t'})));
    p.addParamValue('len',100,@(x)x>0);
    p.addParamValue('debug',false,@islogical);
    p.parse(A,B,varargin{:});
    copulatype = p.Results.copula;
    margintype = p.Results.margin;
    len = p.Results.len;
    debug = p.Results.debug;
    progname = 'ctbmcdiv';
    if (debug)
        fprintf('[%s]: computing MC KL approximation\n',progname);
    end
    sA = ctbsample(A,'margin',margintype,'copula',copulatype,'len',len, 'debug',debug);
    LL1 = ctbll(sA,A,margintype,copulatype);
    LL2 = ctbll(sA,B,margintype,copulatype);
    KL = (LL1-LL2);
end
