# CS159-project ideas
###### tags: `cs159`



> Here are some existing works in this direction that might be of interest:
>- Task Programming is a trajectory generation model that also learns a representation through generation: https://arxiv.org/pdf/2011.13917.pdf and the generative part of task programming is based on this prior work for learning calibratable policies: https://arxiv.org/pdf/1910.01179.pdf
>- Learning Recurrent Hierarchical Representations: https://openreview.net/pdf?id=BkLhzHtlg
>- Another example of learning representations of behavior through generation: https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2.full.pdf
>Some ideas before we chat:
>- Can you learn a representation to generate plausible behavior for a group of N agents given N-1 agents? (And can the representation be used to classify the behavior of the missing agent? The use case would be to understand whether behavior of a single agent can be inferred from others in the group.) We've released a trajectory dataset here with 3 mice / 10 flies that could be used to study this: https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022
>- Can you learn a representation to conditionally generate behaviors given a label (or a mixture of labels)? For example we have labels with attack/sniff for mice (https://data.caltech.edu/records/1991 ), is it possible to generate what the behaviors of the animals would look like on a continuum in that space (10% attack/90% sniff, to 25% attack/75% sniff, etc.)? This can better help us understand the space of behaviors since human annotations are discrete
>- Following up from the above, can you train a model to estimate the likelihood that the animal might display a specific behavior (ex: attack) T seconds into the future? This is important to understand when the animal is committed to doing a certain behavior.



Notes:
1. Given N-1 trajectories, generate the N agents' behavior? Masked Autoencoder?
2. Conditional sequence generative models. Using both labeled and unlabeled data. (Labels need embedding?)
3. Likelihood-based generative models 



# Idea
Can you learn a representation to generate plausible behavior for a group of N agents given N-1 agents? (And can the representation be used to classify the behavior of the missing agent? The use case would be to understand whether behavior of a single agent can be inferred from others in the group.) We've released a trajectory dataset here with 3 mice / 10 flies that could be used to study this: 

1. https://arxiv.org/pdf/2011.13917.pdf
2. https://arxiv.org/abs/2111.06377
3. https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022