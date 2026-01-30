# IDEAS AND TODOS

General Ideas
---

- Use formant changes as augmentation (are datasets biased towards male voices?)

Discriminator Netwerk based on PEAQ. Use it as loss function (https://discuss.pytorch.org/t/train-a-neural-network-and-using-it-as-a-loss-function/156601/9)
---

- annotate data with attributes used for PEAQ?
- Use PESQ as additional label (is speech more important than music for us?)

```python
from asteroid.models import ConvTasNet
import soundfile as sf
import numpy as np
import torch

input, sr = sf.read("mix.wav", dtype="float32")
input = torch.from_numpy(input).unsqueeze(0).requires_grad_()

print(input.grad)

model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k")

output = model(input)
output.backward(torch.ones_like(output))
print(input.grad)
```


