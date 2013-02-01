function [data, testData] = loadDyadicDataAvneesh(dataName, split, useHoldOut)
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
base_data_dir = '~/im3f/data/datasets/';
switch dataName
    case 'ml100k'
        data_dir = [base_data_dir,'/movielens/ml100k/'];
        fname_train = [data_dir, 'u%s.base'];
        fname_test = [data_dir, 'u%s.test'];
        ucol = 1; icol = 2; vcol = 3; % Column index for movielens100k data
	delimiter = '\t';
    case {'ml1m' 'ml10m'}
        data_dir = [base_data_dir,'/movielens/',dataName,'/'];
        fname_train = [data_dir, 'r%s.train'];
        fname_test = [data_dir, 'r%s.test'];
        ucol = 1; icol = 3; vcol = 5; % Column index for movielens1m & 10m data
	delimiter = ':';
    case 'eharmony'
        data_dir = [base_data_dir,'/movielens/',dataName,'/'];
        fname_train = [data_dir, 'r%s.train'];
        fname_test = [data_dir, 'r%s.test'];
        ucol = 1; icol = 3; vcol = 5; 
	%what is delimiter?
    case 'libimseti'
        data_dir = [base_data_dir,'/libimseti/'];
        fname_train = [data_dir, 'r%s.train'];
        fname_test = [data_dir, 'r%s.test'];
        ucol = 1; icol = 2; vcol = 3; 
    case 'dblp'
        data_dir = [base_data_dir, '/dblp/'];
        fname_train = [data_dir, '%s.train'];
        fname_test = [data_dir, '%s.test'];
        ucol = 1; icol = 3; vcol = 5;
        delimiter = ':';
      
end

jaggedCellFile = sprintf([data_dir, 'trainjaggedData%s.mat'], split); 
if exist(jaggedCellFile, 'file')
    % Load jaggedCell exampsBy* matrices from file
    %data = load(trainjaggedCellFile); %testData = load(testjaggedCellFile);
    data = load(jaggedCellFile); 
end

% Load primary training data
%rawData = importdata(sprintf(fname_train, split));
rawData = dlmread(sprintf(fname_train, split), delimiter); 
data.users = uint32(rawData(:,ucol));
data.items = uint32(rawData(:,icol));
data.vals = rawData(:,vcol);
data.numUsers = double(max(data.users));
fprintf('Number of users: %d\n',data.numUsers);
data.numItems = double(max(data.items));
fprintf('Number of items: %d\n',data.numItems);
fprintf('Size of training data: %d user-item pairs\n', size(data.users,1)); 
data.numExamples = length(data.vals); 

% Load test data
clear rawData; 
%rawData = importdata(sprintf(fname_test, split));
rawData = dlmread(sprintf(fname_train, split), delimiter); 
testData.users = uint32(rawData(:,ucol));
testData.items = uint32(rawData(:,icol));
testData.vals = rawData(:,vcol);
testData.numExamples = length(testData.vals); 
clear rawData;

% Cache the data
if ~exist(jaggedCellFile, 'file')
    % Generate jaggedCell exampsBy* matrices and save to file
    data.exampsByUser = jaggedCell(data.users, data.numUsers);
    fprintf('Processed training by user\n');
    data.exampsByItem = jaggedCell(data.items, data.numItems);
    fprintf('Processed training by item\n');
    save(jaggedCellFile, '-STRUCT', 'data', 'exampsByUser', 'exampsByItem');
    fprintf('Saved jagged cell files train\n');
end

%% Obtain test examples by user and by item
testData.exampsByUser = jaggedCell(testData.users, data.numUsers);
fprintf('Processed test data by user\n');
testData.exampsByItem = jaggedCell(testData.items, data.numItems);
fprintf('Processed test data by item\n');

% -----------------------------END OF CODE-------------------------------
