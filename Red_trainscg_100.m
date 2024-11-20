clc; clear all; close all;

% Leer la tabla de datos
BD = readtable('base_datos_cancer_2.csv');

B = 1;
M = 1;

% Inicialización de matrices
Benigno = [];
Maligno = [];

% Separación de datos en Benigno y Maligno
for i = 1:1707
    if BD.Var31(i) == 0
        Benigno(B, :) = BD{i, 1:31};
        B = B + 1;
    else
        Maligno(M, :) = BD{i, 1:31};
        M = M + 1;
    end
end
%////////////////////////
c=1;
c2=1;
for i=1:636
    if (i <637)
        Train_Benigno(c,:)=Benigno(i,:);
        c=c+1;
    else
        Test_Benigno(c2,:)=Benigno(i,:);
        c2 = c2+1;
    end
end
%/////////////////////
c=1;
c2=1;
for i=1:636
    if (i <637)
        Train_Maligno(c,:)=Maligno(i,:);
        c=c+1;
    else
        
        Test_Maligno(c2,:)=Maligno(i,:);
        c2 = c2+1;
    end
end
%///////////////////////////

%Etiquetas de clase

for i=1:636
    T_Maligno(i,:)=[0,1];
end
for i=1:636
    T_Benigno(i,:)=[1,0];
end

input=[Train_Maligno',Train_Benigno'];
Target=[T_Maligno', T_Benigno'];

red= patternnet(10, 'trainscg');

%Organización de hiperparametros
red.trainParam.epochs=(500); %Número de epocas máximas
red.trainParam.max_fail=100; %Verificación mínimos locales
red.trainParam.min_grad=1e-50; %Error máximo permitido,Al llegar aquí, el entrenamiento se detiene
red.trainParam.mu=0.01;%Factor de aprendizaje   
%Configurar entradas y salidas
configure(red,input,Target);

%Funciones de activación de la red
%red.layers{1}.transferFcn='logsig';

%Configuración de la base de datos
red.divideParam.trainRatio=80/100;
red.divideParam.valRatio=10/100;
red.divideParam.testRatio=10/100;

%Entrenamiento de la red
[red,tr]=train(red,input, Target);

%%%%%%%%%%%%%%

%Test y validacion de la red
BDVAL_Benigno= readtable('val_benigno.csv');
BDVAL_Maligno= readtable('val_maligno.csv');
Bien= table2array(BDVAL_Maligno);
Mal= table2array(BDVAL_Benigno);

TP=0;
TN=0;
FN=0; %-> Falso Negativo
FP=0; %-> Falso positivo

%%% Red metricas
for i=1:47
r(i,:)=red(Bien(i,:)'); %Valor 0 1
if (r(i,2)>r(i,1))
TP=TP+1;
else
FP=FP+1;
end
end

for i=1:151
r2(i,:)=red(Mal(i,:)'); %Valor 1 0
if (r2(i,1)>r2(i,2))
TN=TN+1;
else
FN=FN+1;
end
end

% Cálculo de métricas
Exac = ((TP + TN) / (TP + TN + FP + FN)) * 100;
Pre = (TP / (FP + TP)) * 100;
Rec = (TP / (TP + FN)) * 100;
F1 = 2 * ((Pre * Rec) / (Pre + Rec));

% Mostrar resultados
fprintf('Exactitud: %.2f%%\n', Exac);
fprintf('Precisión: %.2f%%\n', Pre);
fprintf('Recall: %.2f%%\n', Rec);
fprintf('F1 Score: %.2f%%\n', F1);
