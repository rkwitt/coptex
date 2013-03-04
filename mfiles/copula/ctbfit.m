function model = ctbfit(data,varargin)
    p = inputParser;
    p.addRequired('data');
    p.addParamValue('margin','Weibull',@(x)any(strcmpi(x,{'Weibull','Gamma','Rayleigh','GGD','Cauchy'})));
    p.addParamValue('copula','Gaussian', @(x)any(strcmpi(x,{'Gaussian','t'})));
    p.addParamValue('debug',false,@islogical);
    p.parse(data,varargin{:});
    margintype = p.Results.margin;
    copulatype = p.Results.copula;
    debug = p.Results.debug;
    progname ='ctbfit';
    dim = size(data,2);
    model = struct('inttrans',[],'margins',[],'Rho',0,'nu',0, 'emp', []);
    if (isempty(data))
        error('data is empty');
    end
    if (debug)
        fprintf('[%s]: fitting %s margins ...\n', progname, margintype);
    end
    for i=1:dim
        col = data(:,i);
        switch margintype
            case 'Weibull'
                param = wblfit(col); % fit Weibull parameters
                model.margins(i,:) = [param(1) param(2)];
                model.inttrans = [model.inttrans wblcdf(col,param(1),param(2))];
                model.emp = [model.emp empiricalCDF(col)];
            case 'Gamma'
                param = gamfit(col);
                model.emp = [model.emp empiricalCDF(col)];
                model.margins(i,:) = [param(1) param(2)];
                model.inttrans = [model.inttrans gamcdf(col,param(1),param(2))];
            case 'GGD'
                [ahat,bhat]=ggdmle(col);
                model.emp = [model.emp empiricalCDF(col)];
                model.margins(i,:) = [ahat bhat];
                model.inttrans = [model.inttrans ggcdf(col,ahat,bhat)];
        end
    end
    if (debug)
        fprintf('[%s]: fitting %s copula\n', ...
            progname, copulatype);
    end
    plow = find(model.inttrans <= 0);
    model.inttrans(plow) = eps;
    phigh = find(model.inttrans >= 1);
    model.inttrans(phigh) = 1-eps;
    if (length(plow) > 0 || length(phigh) >0)
        fprintf('correcting ...\n');
    end
    switch(copulatype)
        case 't'
           [Rho,nu] = copulafit('t',model.inttrans,'Method','ApproximateML');
            model.Rho = Rho;
            model.nu = nu;
        case 'Gaussian'
            Rho = copulafit('Gaussian',model.inttrans);
            model.Rho = Rho;
    end
end
