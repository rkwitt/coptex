function fe = dtcwtfe(data,varargin)
% DTCWTFE Feature extraction from the Dual-Tree Complex Wavelet Transform (DTCWT)
%   FE = DTCWT(DATA,'PARAM1',val1,'PARAM2', val2, ...) extracts a feature
%   representation of the image in the complex wavelet domain.
%   
%   Parameters are:
%
%       DATA                Is a cell array with elements 'image'
%                           'dim', 'idx', 'filename'. 'image' hold the
%                           color or grayscale image. 'dim' specifies the
%                           dimension of the image (in case of color images
%                           'dim' is 3, otherwise 1). 'idx' is not used
%                           here; in case of image classification
%                           it can be used to specify the class the
%                           image belongs to. Finally, 'filename' holds the
%                           image filename.
%         
%       'scales'            maximum decomposition depth of the PWT
%                           (default: 3)
%       
%       'grayscale'         true, if you want to convert the color images in
%                           DATA to grayscale images, where only the
%                           luminance component of the LUV model is kept
%       
%       'preprocess'        true, if you want to subtract the mean from the
%                           image and divide by the std. deviaton (default:
%                           false)
%
%       'debug'             true, if you want verbose debug messages
%                           (default: false)
%
%       'method'            specifies the feature representation to use for
%                           further processing. The output FE is a cell
%                           array with one element 'features' which is in
%                           turn a cell array with SCALES*#SUBBANDS
%                           dimensions. the only exception is the 'rmm'
%                           option where the fitted mixture models are
%                           stored instead. for all other options, the 
%                           total number of extracted features is: 
%                           #subbands * #scales *features/subband. Possible 
%                           feature representations are:
%
%               'energy'        mean and std. deviation of the absolute
%                               DTCWT coefficients values
%               
%               'entropy'       entropy of the absolute coefficient values 
%                               (the coefficients are complex values)
%
%               'wblmle'        Classic direct Weibull ML approach using
%                               moment estimates as starting values
%
%               'wblgmle'       parameters (two) of the Weibull distribution
%                               fitted by using ML and the relation to the
%                               Gumbel distributin (uses Newton-Raphson) -
%                               this is the same approach as the MATLAB
%                               function 'wblfit' implements (although
%                               MATLAB does not use Newton-Raphson)
%
%               'wblmom'        parameters (two) of the Weibull
%                               distribution using moment matching
%                               exploiting the relation to the Gumbel
%                               distribution.
%
%               'gammle'        parameters (two) of the Gamma distribution, 
%                               fitted by the method of maximum likelihood (ML)
%
%               'raymle'        parameters (two) of the Rayleigh
%                               distribution, fitted by the method of
%                               maximum likelihood (has explicit solution).
%                               the mean parameter is always 0 here
%
%               'ggamma'        parameters of a three param GGamma
%                               distribution (fitted by Song08a method)
%                            
%               'gammom'        parameters (two) of the Gamma distribution fitted
%                               by moment matching.
%
%               'rmm'           fits a Rayleigh mixture model with two
%                               components to the absolute coefficient
%                               values of each subband and returns the
%                               fitted mixture model (using EM)
%
%   Author: Roland Kwitt, rkwitt@gmx.at, 2009

    p = inputParser;
    p.addRequired('data',@iscell);
    p.addParamValue('scales',3, @(x)x>=1 && x<= 6);
    p.addParamValue('debug',false,@islogical);
    p.addParamValue('preprocess',false,@islogical);
    p.addParamValue('method','energy', @(x)any(strcmpi(x,{'energy','entropy','ggamma','wblgmle','wblgmom','gammle','gammom','rmm','raymle'})));
    p.addParamValue('colorspace','rgb',@(x)any(strcmpi(x,{'rgb','ybr','hsv','yiq','lab'})));
    p.addParamValue('grayscale',false,@islogical);
    p.parse(data,varargin{:});
    debug = p.Results.debug;
    grayscale = p.Results.grayscale;
    preprocess = p.Results.preprocess;
    scales = p.Results.scales;
    method = p.Results.method;
    colorspace = p.Results.colorspace;
    progname = 'dtcwtfe';
    rgb2lab = makecform('srgb2lab');
    if (debug)
        fprintf('%s: processing %d entries, %d scales using feature %s\n', progname, length(data), scales, method);
    end
    for i=1:length(data)
        if (debug)
            tic;
        end
        im = data{i}.image; 
        if (grayscale)
            im = rgb2gray(im);
        else
            switch colorspace
                case 'lab'
                    im = double(applycform(im,rgb2lab)); 
                case 'rgb'
                    im = double(im);
                case 'ybr'
                    im = double(rgb2ycbcr(im)); % convert to YBR color model
                case 'hsv'
                    im = double(rgb2hsv(im));
                case 'yiq'
                    im = double(rgb2ntsc(im));    
            end
        end
        vec = []; % Feature Vector
        for ch=1:size(im,3)
            channel = double(im(:,:,ch));
            if (preprocess)
                channel = (channel - mean2(channel))./std2(channel);
            end
            [~,Yh] = dtwavexfm2(channel,scales,'near_sym_a','qshift_b');
             for lev=1:scales
                for b=1:6 
                    band = abs(Yh{lev}(:,:,b));
                    band = band(:) + eps; 
                    switch method
                        case 'energy'
							vec = [vec mean(band) std2(band)];									
                        case 'wblmle'
                            param = wblmle(band);
                            vec = [vec param(1) param(2)];
                        case 'entropy'
                            vec = [vec -sum((band(find(band)).^2) ...
                                .*log(band(find(band)).^2))];
                        case 'wblgmom'
                            param = wblgmom(band);
                            vec = [vec param(1), param(2)];
                        case 'raymle'
                            param = raylfit(band);
                            vec = [vec 0 param];
                        case 'ggamma'
                            param = ggammle(band);
                            vec = [vec param];
                        case 'wblgmle'
                            param = wblgmle(band);
                            vec = [vec param(1), param(2)];
                        case 'gammle'
                            param = gamfit(band);
                            vec = [vec param(1) param(2)];
                        case 'gammom'
                            param = gammom(band);
                            vec = [vec param(1) param(2)];
                        case 'rmm'
                            vec{lev,b} = rmm_em(band,'components',2); 
                    end
                end
             end
        end
        fe{i}.features = vec;
        if (debug)
            tim = toc;
            fprintf('%s: processed entry %d in %0.2f [s]\n', progname , i, tim);
        end
    end
end
    








