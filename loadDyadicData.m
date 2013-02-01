function [data, testData] = loadDyadicData(dataName, split, useHoldOut)
%LOADDYADICDATA Load and return dyadic train and test sets as parallel
%array data structures.
%
% Usage:
%    [data, testData] = loadDyadicData(dataName, split)
%    [data, testData] = loadDyadicData(dataName, split, useHoldOut)
%
% Inputs:
%    dataName - Name of dataset to load
%    split - Train-test split of dataset to load
%    useHoldOut - Logical. Load hold-out set in place of test set?
%                 True by default.
%
% Outputs:
%    data - Dyadic dataset structure with parallel array fields:
%           users => Array of uint32 user ids for each example
%           items => Array of uint32 items ids for each example
%           vals => Array of real values for each example

% -----------------------------------------------------------------------     
%
% Last revision: 13-July-2010
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

if nargin < 3
   % By default use hold out set in place of test set if available
   useHoldOut = true;
end

%% Load dyadic data
if(strcmp(dataName, 'movielens100k'))
   if useHoldOut
      warning('loadDyadicData:noHoldOut',... 
	      ['Hold out set unavailable for ',dataName,'. ',...
	       'Loading test set.']);
   end
   jaggedCellFile = sprintf(['movielens100k/' ...
                       'jaggedTest%d.mat'],split);
   if exist(jaggedCellFile, 'file')
      % Load jaggedCell exampsBy* matrices from file
      data = load(jaggedCellFile);
   end

   % Load primary training data
   rawData = importdata(sprintf('movielens100k/u%d.base', split));
   data.users = uint32(rawData(:,1));
   data.items = uint32(rawData(:,2));
   data.vals = rawData(:,3);
   data.numUsers = double(max(data.users));
   data.numItems = double(max(data.items));
   
   % Load test data
   rawData = importdata(sprintf('movielens100k/u%d.test', split));
   testData.users = uint32(rawData(:,1));
   testData.items = uint32(rawData(:,2));
   testData.vals = rawData(:,3);
   clear rawData;

   if ~exist(jaggedCellFile, 'file')
      % Generate jaggedCell exampsBy* matrices and save to file
      data.exampsByUser = jaggedCell(data.users, data.numUsers);
      data.exampsByItem = jaggedCell(data.items, data.numItems);
      save(jaggedCellFile, '-STRUCT', 'data', 'exampsByUser', ...
           'exampsByItem');
   end
end
%% Obtain test examples by user and by item
if ~isfield(testData,'exampsByUser') || isempty(testData.exampsByUser)
    testData.exampsByUser = jaggedCell(testData.users, data.numUsers);
end
if ~isfield(testData,'exampsByItem') || isempty(testData.exampsByItem)
    testData.exampsByItem = jaggedCell(testData.items, data.numItems);
end

% -----------------------------END OF CODE-------------------------------
