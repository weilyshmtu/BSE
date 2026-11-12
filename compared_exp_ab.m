clear
clc
curr_fold = pwd();
addpath("algs\")
addpath("metric\")
addpath("funs\")
addpath("compared algs\")
addpath("datasets\")
datasets = ["Zoo","Yale_32x32","JAFFE_64x64","ORL_32x32","Umist_575_32x32","FERET_32x32","COIL20_32x32","AR_32x32","YALEB_32x32",...
 "COIL100_32x32","MNIST_10k_Train_28x28","pendigits"];
numdatas = length(datasets);
algorithms = ["BSE_MM_P1","BSE_MM_P2", "BSE_MM_P3","LSR","CAN"];
numalgs = length(algorithms);
rng(42)

% --------------------------
% Create results folder if it doesn't exist
% --------------------------
device = "gpu";

normtype = 1;
total_results_dir = fullfile(curr_fold, "TotalResults" + "_" +"gpu" + "_" + num2str(normtype));
if ~exist(total_results_dir, 'dir')
    mkdir(total_results_dir);
end
rn = 2;
for d_ind =length(datasets):length(datasets)
    data = datasets(d_ind);
    
    % load data
    load(data);
    Y = gnd;
    X = fea';

    % normalization
    % unit l2 normalization
    X = normalize_data(X, normtype);
    
    for alg_ind =3 :3
        alg = algorithms(alg_ind);
        
        % options
        opts = [];
        [d, n] = size(X);
        opts.n = n;
        opts.d = d;
        opts.n_clusters = length(unique(Y));
        
        switch alg


            %% BSE: ablation
            case 'BSE_MM_P1'
                %%

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is BSE_MM_GivenS!')
                
                %parameter
                k_set     = [3, 5, 8, 10, 15, 20, 25, 30];
                gamma_set =[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1];

                opts.Dx = L2_distance_1(X, X);
                opts.X = X;
                opts.XtX = X'*X;
                opts.max_iter = 3000;
                opts.tol = 1e-4;
                opts.labels = Y;
                bse_mm_time = [];
                k_init = 10;
                opts.S_init = initialize_S(X, opts.Dx, k_init);
                opts.C_init = (opts.XtX +0.001*eye(n))\opts.XtX;
                % opts.C_init = opts.S_init;
                for a_index = 1 :length(gamma_set)
                    for k_index =  1 : length(k_set)
                        
                        opts.k = k_set(k_index);
                        alpha = gamma_set(a_index)*opts.k^3


                       [C, S, info] = bse_p1_gpu(X, alpha, opts);

                        time = info.time
                        bse_mm_time = [bse_mm_time  time];
                        res = [info.resS info.resC]';
 
                        subplot(1,3,1)
                        plot(1:size(res,2), res(1,:));
                        subplot(1,3,2)
                        plot(1:size(res,2), res(2,:));
                        subplot(1,3,3)
                        obj_hist = info.obj;
                        plot(1:length(obj_hist),obj_hist)
                        drawnow


                        resultC1 = [];
                        resultS1 = [];
                        resultSC1 = [];

                        W_C1 = 0.5*(C' + C);
                        for t=1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end

                        W_S1 = 0.5*(S' + S);
                        for t=1:rn
                            predY = SpectralClustering(W_S1, opts.n_clusters);
                            result21 = Clustering8Measure(Y, predY);
                            resultS1 = [resultS1; result21];
                        end

                       
                        W_SC1 = 0.5*(W_C1+W_S1);
                        for t=1:rn
                            predY = SpectralClustering(W_SC1, opts.n_clusters);
                            result3 = Clustering8Measure(Y, predY);
                            resultSC1 = [resultSC1; result3];
                        end

                        clear W_C1 W_S1 W_SC1
                         

                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, k_index} = mresultC1;
                        resultsC1_std{a_index, k_index} = std(resultC1);

                 
                        mresultS1 = mean(resultS1);
                        resultsS1_mean{a_index, k_index} = mresultS1;
                        resultsS1_std{a_index, k_index} = std(resultS1);



                        mresultSC1 = mean(resultSC1);
                        resultsSC1_mean{a_index, k_index} = mresultSC1;
                        resultsSC1_std{a_index, k_index} = std(resultSC1);


                        acc_C1(a_index, k_index) = mresultC1(1)
                        acc_S1(a_index, k_index) = mresultS1(1)
                        acc_SC1(a_index, k_index) = mresultSC1(1)
     
 
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(k_index);
                        
                        save(fullfile(alg_data_results_dir, filename), "C", "S","resultC1", "resultS1", "resultSC1")

                        clear C S res time predY result11 result21 result12 result22 result3 result4  result5 resultC1 resultS1 mresultC1 ...
                            mresultS1 resultC2 resultS2 mresultC2 mresultS2 mresultSC1 mresultSC2  mresultSC3 mresultSC4
                        pause(0.1);
                    end
                end
                 clc
                best_results = [max(max(acc_C1))  max(max(acc_S1))  max(max(acc_SC1)) ]
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "resultsS1_mean", "resultsS1_std","resultsSC1_mean",...
                    "resultsSC1_std", "bse_mm_time", '-v7.3');
                clear acc_C1 acc_S1 acc_SC1 acc_SC2 acc_SC3 acc_SC4 resultsC1_mean resultsS1_mean resultsS1_std resultsSC1_mean resultsSC1_std resultsSC2_mean resultsSC2_std ...
                    acc_C2 acc_S2 resultsC2_mean resultsS2_mean resultsS2_std  ...
                    resultsSC3_mean resultsSC3_std resultsSC4_mean resultsSC4_std alg_results_dir alg_data_results_dir filename bse_mm_time;

            case 'BSE_MM_P2'
                %%

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is BSE_MM_GivenS!')
                
                %parameter
                k_set     = [3, 5, 8, 10, 15, 20, 25, 30];
                gamma_set =[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1];

                opts.Dx = L2_distance_1(X, X);
                opts.X = X;
                opts.XtX = X'*X;
                opts.max_iter = 3000;
                opts.tol = 1e-4;
                opts.labels = Y;
                bse_mm_time = [];
                k_init = 10;
                opts.S_init = initialize_S(X, opts.Dx, k_init);
                opts.C_init = (opts.XtX +0.001*eye(n))\opts.XtX;
                % opts.C_init = opts.S_init;
                for a_index = 1 :length(gamma_set)
                    for k_index =  1 : length(k_set)
                        
                        opts.k = k_set(k_index);
                        alpha = gamma_set(a_index)*opts.k^3


                       [C, S, info] = bse_p2_gpu(X, alpha, opts);

                        time = info.time
                        bse_mm_time = [bse_mm_time  time];
                        res = [info.resS info.resC]';
 
                        subplot(1,3,1)
                        plot(1:size(res,2), res(1,:));
                        subplot(1,3,2)
                        plot(1:size(res,2), res(2,:));
                        subplot(1,3,3)
                        obj_hist = info.obj;
                        plot(1:length(obj_hist),obj_hist)
                        drawnow


                        resultC1 = [];
                        resultS1 = [];
                        resultSC1 = [];

                        W_C1 = 0.5*(C' + C);
                        for t=1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end

                        W_S1 = 0.5*(S' + S);
                        for t=1:rn
                            predY = SpectralClustering(W_S1, opts.n_clusters);
                            result21 = Clustering8Measure(Y, predY);
                            resultS1 = [resultS1; result21];
                        end

                       
                        W_SC1 = 0.5*(W_C1+W_S1);
                        for t=1:rn
                            predY = SpectralClustering(W_SC1, opts.n_clusters);
                            result3 = Clustering8Measure(Y, predY);
                            resultSC1 = [resultSC1; result3];
                        end

                        clear W_C1 W_S1 W_SC1
                         

                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, k_index} = mresultC1;
                        resultsC1_std{a_index, k_index} = std(resultC1);

                 
                        mresultS1 = mean(resultS1);
                        resultsS1_mean{a_index, k_index} = mresultS1;
                        resultsS1_std{a_index, k_index} = std(resultS1);



                        mresultSC1 = mean(resultSC1);
                        resultsSC1_mean{a_index, k_index} = mresultSC1;
                        resultsSC1_std{a_index, k_index} = std(resultSC1);


                        acc_C1(a_index, k_index) = mresultC1(1)
                        acc_S1(a_index, k_index) = mresultS1(1)
                        acc_SC1(a_index, k_index) = mresultSC1(1)
     
 
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(k_index);
                        
                        save(fullfile(alg_data_results_dir, filename), "C", "S","resultC1", "resultS1", "resultSC1")

                        clear C S res time predY result11 result21 result12 result22 result3 result4  result5 resultC1 resultS1 mresultC1 ...
                            mresultS1 resultC2 resultS2 mresultC2 mresultS2 mresultSC1 mresultSC2  mresultSC3 mresultSC4
                        pause(0.1);
                    end
                end
                 clc
                best_results = [max(max(acc_C1))  max(max(acc_S1))  max(max(acc_SC1)) ]
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "resultsS1_mean", "resultsS1_std","resultsSC1_mean",...
                    "resultsSC1_std", "bse_mm_time", '-v7.3');
                clear acc_C1 acc_S1 acc_SC1 acc_SC2 acc_SC3 acc_SC4 resultsC1_mean resultsS1_mean resultsS1_std resultsSC1_mean resultsSC1_std resultsSC2_mean resultsSC2_std ...
                    acc_C2 acc_S2 resultsC2_mean resultsS2_mean resultsS2_std  ...
                    resultsSC3_mean resultsSC3_std resultsSC4_mean resultsSC4_std alg_results_dir alg_data_results_dir filename bse_mm_time;
    
            case 'BSE_MM_P3'
                %%

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is BSE_MM_GivenC!')
                
                %parameter
               k_set     = [3, 5, 8, 10, 15, 20, 25, 30];
                gamma_set =[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1];

                opts.Dx = L2_distance_1(X, X);
                opts.X = X;
                opts.XtX = X'*X;
                opts.max_iter = 3000;
                opts.tol = 1e-4;
                opts.labels = Y;
                bse_mm_time = [];
                k_init = 10;
                opts.S_init = initialize_S(X, opts.Dx, k_init);
                opts.C_init = (opts.XtX +0.001*eye(n))\opts.XtX;
                % opts.C_init = opts.S_init;
                for a_index = 1 :length(gamma_set)
                    for k_index =  1 : length(k_set)
                        
                         opts.k = k_set(k_index);
                        alpha = gamma_set(a_index)*opts.k^3

                    
                        [C, S, info] = bse_p3_gpu(X, alpha, opts);


                        time = info.time
                        bse_mm_time = [bse_mm_time  time];
                        res = [info.resS info.resC]';
 
                        subplot(1,3,1)
                        plot(1:size(res,2), res(1,:));
                        subplot(1,3,2)
                        plot(1:size(res,2), res(2,:));
                        subplot(1,3,3)
                        obj_hist = info.obj;
                        plot(1:length(obj_hist),obj_hist)
                        drawnow


                        resultC1 = [];
                        resultS1 = [];
                        resultSC1 = [];

                        W_C1 = 0.5*(C' + C);
                        for t=1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end

                        W_S1 = 0.5*(S' + S);
                        for t=1:rn
                            predY = SpectralClustering(W_S1, opts.n_clusters);
                            result21 = Clustering8Measure(Y, predY);
                            resultS1 = [resultS1; result21];
                        end

                       
                        W_SC1 = 0.5*(W_C1+W_S1);
                        for t=1:rn
                            predY = SpectralClustering(W_SC1, opts.n_clusters);
                            result3 = Clustering8Measure(Y, predY);
                            resultSC1 = [resultSC1; result3];
                        end

                        clear W_C1 W_S1 W_SC1
                         

                        mresultC1 = mean(resultC1);
                        resultsC1_mean{a_index, k_index} = mresultC1;
                        resultsC1_std{a_index, k_index} = std(resultC1);

                 
                        mresultS1 = mean(resultS1);
                        resultsS1_mean{a_index, k_index} = mresultS1;
                        resultsS1_std{a_index, k_index} = std(resultS1);



                        mresultSC1 = mean(resultSC1);
                        resultsSC1_mean{a_index, k_index} = mresultSC1;
                        resultsSC1_std{a_index, k_index} = std(resultSC1);


                        acc_C1(a_index, k_index) = mresultC1(1)
                        acc_S1(a_index, k_index) = mresultS1(1)
                        acc_SC1(a_index, k_index) = mresultSC1(1)
     
 
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" + num2str(k_index);
                        
                        save(fullfile(alg_data_results_dir, filename), "C", "S","resultC1", "resultS1", "resultSC1")

                        clear C S res time predY result11 result21 result12 result22 result3 result4  result5 resultC1 resultS1 mresultC1 ...
                            mresultS1 resultC2 resultS2 mresultC2 mresultS2 mresultSC1 mresultSC2  mresultSC3 mresultSC4
                        pause(0.1);
                    end
                end
                clc
                best_results = [max(max(acc_C1))  max(max(acc_S1))  max(max(acc_SC1)) ]
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "resultsS1_mean", "resultsS1_std","resultsSC1_mean",...
                    "resultsSC1_std", "bse_mm_time", '-v7.3');
                clear acc_C1 acc_S1 acc_SC1 acc_SC2 acc_SC3 acc_SC4 resultsC1_mean resultsS1_mean resultsS1_std resultsSC1_mean resultsSC1_std resultsSC2_mean resultsSC2_std ...
                    acc_C2 acc_S2 resultsC2_mean resultsS2_mean resultsS2_std  ...
                    resultsSC3_mean resultsSC3_std resultsSC4_mean resultsSC4_std alg_results_dir alg_data_results_dir filename bse_mm_time;
    
            case 'LSR'
                %%

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is LSR!')
                
                %parameter
                alpha_set = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
                opts.Dx = L2_distance_1(X, X);
                opts.X = X;
                opts.XtX = X'*X;
                opts.max_iter = 3000;
                opts.tol = 1e-4;
                opts.labels = Y;
                lsr_time = [];
               
                for a_index = 1:length(alpha_set)
                        [C, info] = lsr_gpu(X, alpha_set(a_index), opts);
                        lsr_time = [lsr_time info.time];

                        resultC1 = [];
                        W_C1 = 0.5*(C' + C);
                        for t=1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end

                        clear W_C1 W_S1 W_SC1
                         
                        mresultC1 = max(resultC1);
                        resultsC1_mean{a_index} = mresultC1;
                        resultsC1_std{a_index} = std(resultC1);

                        acc_C1(a_index) = mresultC1(1);
                  
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename = num2str(a_index) + "x" ;
                        
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1")

                        clear C S res time predY result11 result21 result12 result22 result3 result4  result5 resultC1 resultS1 mresultC1 ...
                            mresultS1 resultC2 resultS2 mresultC2 mresultS2 mresultSC1 mresultSC2  mresultSC3 mresultSC4
                        pause(0.1);
                end
                clc
                best_results = [max(max(acc_C1))]
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "lsr_time", '-v7.3');
                clear acc_C1 acc_S1 acc_SC1 acc_SC2 acc_SC3 acc_SC4 resultsC1_mean resultsS1_mean resultsS1_std resultsSC1_mean resultsSC1_std resultsSC2_mean resultsSC2_std ...
                    acc_C2 acc_S2 resultsC2_mean resultsS2_mean resultsS2_std  ...
                    resultsSC3_mean resultsSC3_std resultsSC4_mean resultsSC4_std alg_results_dir alg_data_results_dir filename lsr_time;
    
            case 'CAN'
                %%

                alg_results_dir = fullfile(total_results_dir, alg);
                if ~exist(alg_results_dir, 'dir')
                    mkdir(alg_results_dir);
                end
                
                fprintf("Processing on "+ data + " dataset!\n")
                disp('Running algorithm is CAN!')
                
                %parameter
                k_set     = [3, 5, 8, 10, 15, 20, 25, 30];
            

                opts.Dx = L2_distance_1(X, X);
                opts.X = X;
                opts.XtX = X'*X;
                opts.max_iter = 3000;
                opts.tol = 1e-4;
                opts.labels = Y;
                can_time = [];
                
                % opts.C_init = opts.S_init;
               for k_index = 1:length(k_set)
                        [C, info] = can_gpu(X, k_set(k_index), opts);
                        can_time = [can_time info.time];
                        W_C1 = 0.5*(C' + C);
                        resultC1 = [];
                        for t=1:rn
                            predY = SpectralClustering(W_C1, opts.n_clusters);
                            result11 = Clustering8Measure(Y, predY);
                            resultC1 = [resultC1; result11];
                        end

                  
                        clear W_C1 W_S1 W_SC1
                         

                        mresultC1 = max(resultC1);
                        resultsC1_mean{ k_index} = mresultC1;
                        resultsC1_std{ k_index} = std(resultC1);

                 


                        acc_C1( k_index) = mresultC1(1);
                  
                        alg_data_results_dir = fullfile(alg_results_dir, data);
                        if ~exist(alg_data_results_dir, 'dir')
                            mkdir(alg_data_results_dir);
                        end
                        filename =  num2str(k_index)+ "x";
                        
                        save(fullfile(alg_data_results_dir, filename), "C", "resultC1")

                        clear C S res time predY result11 result21 result12 result22 result3 result4  result5 resultC1 resultS1 mresultC1 ...
                            mresultS1 resultC2 resultS2 mresultC2 mresultS2 mresultSC1 mresultSC2  mresultSC3 mresultSC4
                        pause(0.1);
                end
                clc
                best_results = [max(max(acc_C1))]
                save(fullfile(alg_results_dir, data), "resultsC1_mean", "resultsC1_std", "can_time", '-v7.3');
                clear acc_C1 acc_S1 acc_SC1 acc_SC2 acc_SC3 acc_SC4 resultsC1_mean resultsS1_mean resultsS1_std resultsSC1_mean resultsSC1_std resultsSC2_mean resultsSC2_std ...
                    acc_C2 acc_S2 resultsC2_mean resultsS2_mean resultsS2_std  ...
                    resultsSC3_mean resultsSC3_std resultsSC4_mean resultsSC4_std alg_results_dir alg_data_results_dir filename can_time;
        end % switch
         
    end % for alg_ind
end % for d_ind