function data = ctbgen(image,varargin)
    p = inputParser;
    p.addRequired('image',@(x)~isempty(x));
    p.addParamValue('margin','Weibull',@(x)any(strcmpi(x,{'Weibull','Gamma','Rayleigh','GGD','Cauchy'})));
    p.addParamValue('colorspace','ybr',@(x)any(strcmpi(x,{'rgb','ybr','hsv','yiq','lab'})));
    p.addParamValue('levels',3,@(x)x>=1 && x <= 5);
    p.addParamValue('step',1,@(x)x>=1 && x <= 8);
    p.addParamValue('debug',false,@islogical);
    p.addParamValue('grayscale',false,@islogical);
    p.parse(image,varargin{:});
    type = p.Results.margin;
    colorspace =  p.Results.colorspace;
    levels = p.Results.levels;
    step = p.Results.step;
    grayscale = p.Results.grayscale;
    debug = p.Results.debug;
    dim = size(image,3);
    progname ='ctbgen';
    rgb2lab = makecform('srgb2lab');
    
    if (debug)
        fprintf('[%s]: margin = %s, levels = %d\n', progname, type,levels);
    end
    
    if (grayscale && dim > 1)
        if (debug)
            fprintf('converting to grayscale\n');
        end
        image = rgb2gray(image);
        dim = 1;
    end
    if (dim > 1) 
        % we assume input images with dim>1 are RGB images ;) otherwise
        % it is a grayscale image and we do not care about color space
        % conversions;
        fprintf('[%s]: convert to %s colorspace\n', progname, upper(colorspace));
        switch colorspace
            case 'lab'
                image = double(applycform(image,rgb2lab)); 
            case 'rgb'
                image = double(image);
            case 'ybr'
                image = double(rgb2ycbcr(image)); % convert to YBR color model
            case 'hsv'
                image = double(rgb2hsv(image));
            case 'yiq'
                image = double(rgb2ntsc(image));
        end
    end
    for i=1:dim
		plane = double(image(:,:,i));
        plane = (plane - mean2(plane)./std2(plane));
        switch type
            case {'Weibull','Gamma','Rayleigh'}
                [Yl,Yh] = dtwavexfm2(plane,levels,'near_sym_b','qshift_b');
                channels{i}.Yh = Yh;
                channels{i}.Yl = Yl;
            case {'GGD','Cauchy'}
                dwtmode('per','nodisp');
                [c,s] = wavedec2(plane,levels,'bior4.4');
                channels{i}.c = c;
                channels{i}.s = s;
        end
    end
    for l=1:levels
        data{l} = [];
        switch type
            case {'Weibull','Gamma','Rayleigh'}
                for or=1:6
                    for ch =1:dim 
                        coef = abs(channels{ch}.Yh{l}(:,:,or));
                        coef = coef(:)+eps;
                        coef = coef(1:step:end);
                        data{l} = [data{l} coef];
                    end
                end
            case {'GGD','Cauchy'}
                for ch=1:dim
                    [H,V,D] = detcoef2('all',channels{ch}.c,channels{ch}.s,l);
                    data{l} = [data{l} H(:) V(:) D(:)];
                end
        end
    end
end





