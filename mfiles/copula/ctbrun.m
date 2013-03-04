
function varargout = ctbrun(data,varargin)
    p = inputParser;
    p.addRequired('data',@iscell);
    p.addParamValue('models',{},@iscell); 
    p.addParamValue('stage','genmodel',@(x)any(strcmpi(x,{'genmodel','runsim','runll'})));
    p.addParamValue('margin', 'Weibull', @(x)any(strcmpi(x,{'Weibull','Gamma', 'GGD'})));
    p.addParamValue('colorspace','ybr',@(x)any(strcmpi(x,{'rgb','ybr','hsv','yiq','lab'})));
    p.addParamValue('copula', 'Gaussian', @(x)any(strcmpi(x,{'Gaussian','t'})));
    p.addParamValue('grayscale',false,@islogical);
    p.addParamValue('level',3,@(x)x>=1);
    p.addParamValue('step',1,@(x)x>=1 && x <= 8);
    p.addParamValue('samples',512,@(x)x>=100);
    p.addParamValue('debug',false,@islogical);
    p.parse(data,varargin{:});
    debug = p.Results.debug;
    models = p.Results.models;
    grayscale = p.Results.grayscale;
    level = p.Results.level;
    step = p.Results.step;
    samples = p.Results.samples;
    stage = p.Results.stage;
    margintype = p.Results.margin;
    copulatype = p.Results.copula;
    colorspace = p.Results.colorspace;
    progname ='ctbrun';
    if (debug)
        fprintf('[%s]: running %s for %s margins and %s copula (level %d)\n', ...
            progname, stage, margintype, copulatype, level);
    end
    switch stage
        case 'genmodel'
            if (~isempty(models))
                error('model parameter given although stage is genmodel');
            end
            for i=1:length(data)
                fprintf('processing image %s\n', data{i}.filename);   
                X = ctbgen(data{i}.image,'margin',margintype,'levels',level,'debug',debug,'step',step,'colorspace',colorspace,'grayscale',grayscale);
                model{i} = ctbfit(X{level},'margin', margintype, 'copula',copulatype,'debug',debug);
                model{i}.X = X{level};
            end
            varargout(1) = {model};
        case 'runsim'
            for i=1:length(data)
                fprintf('processing image %s\n', data{i}.filename);
                for j=1:i
                    div(i,j) = ctbmcdiv(models{i},models{j},...
                        'margin','Weibull', ...
                        'copula','Gaussian','len',samples,'debug',debug);
                end
            end
            varargout(1) = {div};
        case 'runll'
            for i=1:length(data)
                X = ctbgen(data{i}.image,'margin',margintype,'levels',level,'debug',debug,'step',step,'colorspace',colorspace);
                fprintf('processing image %s\n', data{i}.filename);
                for j=1:length(data)
                    div(i,j) = ctbll(X{level},models{j},margintype, copulatype); 
                end
            end    
            varargout(1) = {div};
    end
end        

