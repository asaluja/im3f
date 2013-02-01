function make_mex(makeCommon, make_m3f_tib, make_m3f_tif, mexOpts)
% MAKE_MEX Compile and link MEX file dependencies.
%
% Usage:
%    make_mex(makeCommon, make_m3f_tib, make_m3f_tif, mexOpts) 
%
% Inputs:
%    makeCommon - Logical. Make MEX files common to all models?
%    makem3f_tib - Logical. Make m3f_tib MEX files?
%    makem3f_tif - Logical. Make m3f_tif MEX files?
%    mexOpts - (Optional) Cell array of string options to be passed to mex function
%
% Notes:
%    If you don't specify mexOpts, this function will provide the necessary
%    arguments (CFLAGS and linking) to use C99 mode, OpenMP, and link to
%    the GNU OMP (lgomp) and Gnu Scientific Library (lgsl, lgslcblas) which
%    are required for this program to work. If this doesn't work for you,
%    and you have all the prerequisities installed, try setting mexOpts to
%    empty {} and set the necessary options in your mexopts.sh file.

% -----------------------------------------------------------------------     
%
% Last revision: 12-July-2010
%
% Authors: Lester Mackey and David Weiss
% License: MIT License
%
% Copyright (c) 2010 Lester Mackey & David Weiss
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------

% -----------------------------BEGIN CODE--------------------------------

if nargin == 0
    make_m3f_tif = 0;
    make_m3f_tib = 1;
    makeCommon = 1;
end

if nargin < 4
    % enable debugging:
    % mexOpts = {'CFLAGS=\$CFLAGS -std=c99 -fopenmp', ...    
    %            '-lgomp', '-lgslcblas', '-lmwlapack', '-lmwblas', '-lgsl', '-lm', ...
    %            '-g'};
    % disable debugging:
    % mexOpts = {'CFLAGS=\$CFLAGS -std=c99 -fopenmp -lgomp', ...    % Original
     
    %'-lgomp', '-lgsl', '-lgslcblas', '-lmwlapack', '-lmwblas',  '-lm', ...
    mexOpts = {'CFLAGS=\$CFLAGS -std=c99 -fopenmp', ...    % Removed openmp                   
                 '-lgomp', '-lgsl', '-lgslcblas', '-lmwlapack', '-lmwblas',  '-lm', ...
     '-g'}; % Debug
            %mexOpts = {'CFLAGS=\$CFLAGS -std=c99 -fopenmp', ...    % Removed openmp                   
            %     '-lgomp', '-lgsl', '-lgslcblas', '-lmwlapack', '-lmwblas',  '-lm', ...
            %'-DNDEBUG'};
end

display(['Compiling mex files with ', strcat(mexOpts{:})]);
if(makeCommon)
   fprintf('Compiling common files...\n');
   mex('mex/accumarrayMex.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/sampleVectorMex.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/sampleTopicParams.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/seedMexRand.c', 'mex/mexCommon.c', mexOpts{:});
end

if(make_m3f_tib)
   fprintf('Compiling m3f_tib files...\n');
   mex('mex/sgdFactorVectors.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tib_predictMex.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tib_predictToneyMex.c', 'mex/mexCommon.c',mexOpts{:});
   mex('mex/getResidualMex.c', 'mex/mexCommon.c', mexOpts{:}); 
   mex('mex/sampleCRFTablesMex.c', 'mex/mexCommon.c', mexOpts{:}); 
   mex('mex/sampleCRFDishsMex.c', 'mex/mexCommon.c', mexOpts{:}); 
   mex('mex/m3f_tib_sampleFactorVectorsCRF.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tib_sampleOffsets.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tib_sampleTopics.c', 'mex/mexCommon.c', mexOpts{:});
end

if(make_m3f_tif)
   fprintf('Compiling m3f_tif files...\n');
   mex('mex/sgdFactorVectors.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tif_predictMex.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tif_sampleFactorVectors.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tif_sampleTopicFactorVectors.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tif_sampleTopics.c', 'mex/mexCommon.c', mexOpts{:});
   mex('mex/m3f_tif_sampleBiases.c', 'mex/mexCommon.c', mexOpts{:});
end

% -----------------------------END OF CODE-------------------------------
