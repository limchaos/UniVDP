# Third-party notices

## IQA-PyTorch (TOPIQ / CFANet)

`pyunivdp/univdp_net.py` adapts the CFANet attention head of TOPIQ from IQA-PyTorch (https://github.com/chaofengc/IQA-PyTorch, file `pyiqa/archs/topiq_arch.py`):

> C. Chen, J. Mo, J. Hou, H. Wu, L. Liao, W. Sun, Q. Yan, W. Lin. TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment. IEEE Transactions on Image Processing, 2024.

The IQA-PyTorch code is distributed under the following license:

```
S-Lab License 1.0

Copyright 2022 S-Lab

Redistribution and use for non-commercial purpose in source and 
binary forms, with or without modification, are permitted provided 
that the following conditions are met:

1. Redistributions of source code must retain the above copyright 
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright 
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.

3. Neither the name of the copyright holder nor the names of its 
   contributors may be used to endorse or promote products derived 
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In the event that redistribution and/or use for commercial purpose in 
source or binary forms, with or without modification is required, 
please contact the contributor(s) of the work.
```

## DINOv3

The checkpoint contains DINOv3 ViT-S/16 weights (Meta AI, https://github.com/facebookresearch/dinov3), blocks 0 to 7, fine-tuned with UniVDP. The network definition comes from timm (Apache 2.0). Use of the weights is subject to the DINOv3 License.

## ColorVideoVDP

UniVDP calls ColorVideoVDP (https://github.com/gfxdisp/ColorVideoVDP, v0.5.6) for the display model, video reading and the visibility statistics. It is installed as a dependency and not distributed with this repository; see its license.
