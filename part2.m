global Weights1 Weights2 Weights3 Biases_hidden output_bias Y X learning_rate number_of_inputs number_of_hidden Weights1_connections useBias
%======================================ADJUST NETWORK HERE======================================%.
%1. Tweak Hyperparameters
number_of_training_set=2;
number_of_inputs = 2;
number_of_hidden = 3;
learning_rate = 1.0;
useBias = false;
num_of_epochs=1;

%2. Please add input and targets here
X = [];%Add inputs
Y = [];%Add target
%You can choose to randomise inputs and targets by enabling this
generateRandomDataset = false;

%2.1 Weight to remove using [x y] which corresponds to an indece in Weights 1. e.g [1 1; 2 1] will remove first row and second row first weights
weights_to_remove = []; 

%3. Will use the example network
training_example = true; 
%===============================================================================================%.

if ~training_example
    %Intitialise weights to zero
    Weights1 = zeros(number_of_inputs,number_of_hidden);
    Weights2 = zeros(1,number_of_inputs);
    Weights3 = zeros(1,number_of_hidden);
    if generateRandomDataset
        %Randomly add input and target val between 0 & 1
        X = randi([0, 1], [number_of_training_set,number_of_inputs]);
        Y = randi([0, 1], [number_of_training_set,1]);
    end
    %Create weights and initialise them with random weight values between
    min = -0.5;max=0.5;
    %Weights from i -> j
    Weights1=generateRandomWeights(Weights1,min,max);
    %Weights from i -> k
    Weights2=generateRandomWeights(Weights2,min,max);
    %Weights from j -> k
    Weights3=generateRandomWeights(Weights3,min,max);
    %Datastructure used for adding/removing weights between i->j. 
    %1 = Weight exists
    %0 = Weight does not exist
    Weights1_connections = ones(number_of_inputs, number_of_hidden);
    for i=1:size(weights_to_remove,1)
        removeWeight(weights_to_remove(i,1),weights_to_remove(i,2));
    end
    

else
    %This will use the example weights, inputs and targets
    number_of_training_set=2;
    number_of_inputs = 2;
    number_of_hidden = 3;
    X = [1 0;0 1];
    Y = [1;1];
    Weights1 = [0.2 0.3 0;0 -0.15 0.25];
    Weights1_connections = [1 1 0;0 1 1];%Adding and removing some weights
    Weights2 = [0.15 -0.1]; 
    Weights3 = [-0.1 0.2 -0.15]; 
end


%Biases for hidden and output
Biases_hidden = zeros(1,number_of_hidden);
output_bias = 0;

%======================================TRAINING======================================%.
for epoch = 1 :num_of_epochs
    for index=1:number_of_training_set
        %Get the current training example and output
        x = X(index,:);y = Y(index);
        %Forward propagation
        [Hidden,out] = forwardPropogate(x);
        error = y-out;%Get the error
        loss = sum(sum( error.^2 ));%The loss
        %Backward propagation
        [Weights1_delta,Weights2_delta,Weights3_delta,Biases_hidden_deltas,output_bias_delta]=backpropogate(x,y,out,Hidden);
        %Update the weights
        for i=1:number_of_inputs
            %Update weights input to output
            Weights2(i) = Weights2(i) + Weights2_delta(i); 
            for j=1:number_of_hidden 
               if(if_weight_exist(i,j)) %Only updates if weight exists
                %Update weights input to hidden
                Weights1(i,j) = Weights1(i,j) + Weights1_delta(i,j);
               end
            end
        end
        %Update weights hidden to output
        for i=1:number_of_hidden
            Weights3(i) = Weights3(i) + Weights3_delta(i); 
        end
        
        %Update the bias
        output_bias = output_bias+output_bias_delta; %Update output bias
        for i=1:number_of_hidden
            Biases_hidden(i) = Biases_hidden(i)+ Biases_hidden_deltas(i); %Update hidden bias
        end
    end    
    %Output the epoch number and accuracy
    fprintf('Epoch %3d:  Loss = %f\n',epoch,loss);
end


%Output the final weights
disp("Final Weights")
disp("---WEIGHTS Input to Hidden---")
disp(round(Weights1,4))
disp("---WEIGHTS Input to Output---")
disp(round(Weights2,4))
disp("---WEIGHTS Hidden to Output---")
disp(round(Weights3,4))

if training_example
    disp("w1 = "+round(Weights1(1,1),4))
    disp("w2 = "+round(Weights2(1),4))
    disp("w3 = "+round(Weights1(1,2),4))
    disp("w4 = "+round(Weights1(2,2),4))
    disp("w5 = "+round(Weights2(2),4))
    disp("w6 = "+round(Weights1(2,3),4))
    disp("w7 = "+round(Weights3(3),4))
    disp("w8 = "+round(Weights3(2),4))
    disp("w9 = "+round(Weights3(1),4)) 
end


function [Hidden,out] = forwardPropogate(x)
    global Weights1 Weights2 Weights3 Biases_hidden output_bias number_of_hidden number_of_inputs
    Hidden = [zeros(1,number_of_hidden)];
    for j=1:number_of_hidden
        sum =0;
        for i=1:number_of_inputs
            sum = sum+(Weights1(i,j) * x(i));
        end    
        sum = sigmoid(sum+Biases_hidden(j));
        Hidden(j) = sum;
    end
    Inputs2 = [Hidden x];
    Weights = [Weights3 Weights2];
    out = 0;    
    for i=1:size(Weights,2)
        out = out + (Weights(i)*Inputs2(i));
    end
    out = out+output_bias;    
end

function [Weights1_delta,Weights2_delta,Weights3_delta, Biases_hidden_deltas, output_bias_delta]=backpropogate(x,y,out,Hidden)
    global Weights2 Weights3 learning_rate number_of_hidden number_of_inputs Weights1 Biases_hidden useBias
        %Compute the benefit at node K
        output_benefit = y - out;
        %Compute the changes for weights j->k
        Weights3_delta = zeros(size(Weights3));
        output_bias_delta =0;
        for i=1:number_of_hidden
            Weights3_delta(i)= learning_rate * output_benefit * Hidden(i);
        end        
        %Compute the change for output biases
        if useBias
            output_bias_delta = learning_rate * output_benefit;
        end

        %Compute the changes for weights i->k
        Weights2_delta = zeros(size(Weights2));
        for i=1:number_of_inputs
            Weights2_delta(i)= learning_rate * output_benefit * x(i);
        end 
        
        %Compute the changes for weights i->j
        Weights1_delta = zeros(size(Weights1));
        Biases_hidden_deltas = zeros(size(Biases_hidden));
        for i=1:number_of_inputs
            for j=1:number_of_hidden
                Hidden_beta = (Hidden(j) .* (1 - Hidden(j)) *(output_benefit * Weights3(j)));
                Weights1_delta(i,j) =learning_rate* x(i) * Hidden_beta;
                
                %Compute the change for hidden biases
                if useBias 
                    Biases_hidden_deltas(j) = learning_rate * Hidden_beta;
                end
            end
        end     
end

%======================================HELPER FUNCTIONS======================================%.

function Weights = generateRandomWeights(Weights,min,max)
    a = min;
    b = max;
    for i=1:size(Weights,1)
        for j=1:size(Weights,2)
            Weights(i,j) = a + (b-a).*rand(1);
        end
    end

end

function res = if_weight_exist(i,j)
    global Weights1_connections
    res = Weights1_connections(i,j)~=0;
end

function removeWeight(x,y)
    global Weights1_connections Weights1
    Weights1_connections(x,y)=0;
    Weights1(x,y)=0;
end

function sigmoid = sigmoid(val)
    sigmoid = 1.0 ./( 1.0 + exp( -val ));
end