# Tese Mestrado Eng. Biomédica Guilherme Miranda 2022

## Codificação Neural e Plasticidade Sináptica em Modelos Animais de Doenças do Neurodesenvolvimento

## Resumo
#### A neurofibromatose tipo 1 (NF1) é uma doença hereditária associada a perturbações do desenvolvimento neurológico sendo um dos impactos desta doença as alterações da neuroplasticidade, incluindo a do córtex visual. Surge então a necessidade de aferir a integridade da neuroplasticidade, através dos potenciais evocados visuais (VEP) sob a forma de potenciação da resposta seletiva a estímulos (SRP). A SRP está presente nos VEP quando o potencial obtido tem maior intensidade num estímulo familiar do que num estímulo novel. Assim, o principal objetivo deste trabalho foi a classificação de estímulos novel e familiar presentes no eletroencefalograma (EEG), após cada estímulo. Foram então estudados métodos que conseguissem extrair do sinal EEG características diferenciadoras de cada tipo de estímulo que permitissem a sua classificação. Para atingir este objetivo foi então desenvolvido um programa com recurso a técnicas de Machine Learning que dado um período temporal após cada estímulo fosse capaz de classificar o estímulo que resultou na resposta presente no período temporal, como novel ou familiar. Com o presente trabalho, foi possível concluir que é possível um algoritmo de machine learning classificar corretamente cada estímulo, uma vez que pelos resultados obtidos os valores da accuracy são bastante bons e o modelo apresenta robustez.

## Objetivos
1. Perceber a possibilidade da utilização de técnicas de machine learning para a correta classificação do tipo de estímulo presente no eletroencefalograma em estudo.
2. Desenvolver um programa capaz de realizar o processamento de sinal, a extração de features representativas de cada estímulo e a sua classificação.


## Métodos
1. dataVisualization -> usada para visualizar a distribuição espacial dos dados com o t-SNE

2. featureExtraction -> processo de extração de features

3. featureReduction -> redução de features com o truncated SVD

4. graphs -> criação dos gráficos usados na tese

5. models -> definição dos modelos de machine learning

6. utils -> métodos auxiliares, como a pipeline e leitura de ficheiros

7. VP -> extração das epochs


