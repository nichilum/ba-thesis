// Note:
//   1. *paragraph:* What is the motivation of your thesis? Why is it interesting from a scientific point of view? Which main problem do you like to solve?
//   2. *paragraph:* What is the purpose of the document? What is the main content, the main contribution?
//   3. *paragraph:* What is your methodology? How do you proceed?

Reverberation is an inherent characteristic of real-world recording environments, and its removal (dereverberation) poses a significant challenge due to the highly non-linear, time-dispersive nature of room acoustics. While deep learning has enabled substantial progress in speech dereverberation, the generalization of these models to diverse audio signals is not well documented. In recent years metrics such as the @SI-SNR:short and @PESQ:short have been successfully utilized as loss functions. Their performance for dereverberation has not been systematically evaluated.

This thesis evaluates two state-of-the-art speech dereverberation models (Conv-TasNet and StoRM) for out-of-domain performance on diverse audio signals. Based on this evaluation, a novel dereverberation network was developed, built on the architecture of Conv-TasNet. Furthermore, the applicability of six fundamentally distinct loss functions for dereverberation is analyzed, concluding in the implementation of a novel scoring network.

Our results show a gap between speech and out-of-domain dereverberation performance in the investigated existing models. The evaluation of loss functions reports the @SI-SNR:short as the strongest metric. A scoring network was successfully implemented and validated, showing strong predictive performance, outperforming the @SI-SNR:short as a dereverberation indicator. While the @SI-SNR:short was successfully used in the implementation of our dereverberation network, the scoring network failed to generate usable results.
