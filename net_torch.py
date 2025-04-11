import torch
import torch.nn as nn
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            dx_ll, dx_lh, dx_hl, dx_hh = dx[:, :C], dx[:, C:C * 2], dx[:, C * 2:C * 3], dx[:, C * 3:]

            dx_x_ll = torch.nn.functional.conv_transpose2d(dx_ll, w_ll.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_lh = torch.nn.functional.conv_transpose2d(dx_lh, w_lh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hl = torch.nn.functional.conv_transpose2d(dx_hl, w_hl.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hh = torch.nn.functional.conv_transpose2d(dx_hh, w_hh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            return dx_x_ll + dx_x_lh + dx_x_hl + dx_x_hh, None, None, None, None
        else:
            return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self):
        super(DWT_2D, self).__init__()
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)

        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)

        self.w_ll = w_ll.to(dtype=torch.float32)
        self.w_lh = w_lh.to(dtype=torch.float32)
        self.w_hl = w_hl.to(dtype=torch.float32)
        self.w_hh = w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        _, C, _, _ = x.shape
        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        x_ll, x_lh, x_hl, x_hh = x[:, :C // 4], x[:, C // 4:C * 2 // 4], x[:, C * 2 // 4:C * 3 // 4], x[:, C * 3 // 4:]
        x_1_ll = torch.nn.functional.conv_transpose2d(x_ll, w_ll.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_lh = torch.nn.functional.conv_transpose2d(x_lh, w_lh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hl = torch.nn.functional.conv_transpose2d(x_hl, w_hl.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hh = torch.nn.functional.conv_transpose2d(x_hh, w_hh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        return x_1_ll + x_1_lh + x_1_hl + x_1_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            _, C, _, _ = ctx.shape
            C //= 4

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self):
        super(IDWT_2D, self).__init__()
        w_ll = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32, requires_grad=False)

        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = filters

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class raise_channel(nn.Module):
    def __init__(self, in_channel, target_channel):
        super(raise_channel, self).__init__()
        self.raise_conv = nn.Sequential(
            nn.Conv2d(in_channel, target_channel, 5, 1, 2, bias=True),
            nn.PReLU(num_parameters=target_channel, init=0.01),
            nn.Conv2d(target_channel, target_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        x = self.raise_conv(x)
        return x


class reduce_channel(nn.Module):
    def __init__(self, ms_target_channel, L_up_channel):
        super(reduce_channel, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(ms_target_channel, ms_target_channel, 3, 1, 1, bias=True),
            nn.PReLU(num_parameters=ms_target_channel, init=0.01),
            nn.Conv2d(ms_target_channel, L_up_channel, 3, 1, 1, bias=True),
            nn.Conv2d(L_up_channel, L_up_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.reduce_conv(x)


class FFN(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(FFN, self).__init__()
        self.FFN_channel, self.out_channel = FFN_channel, out_channel
        self.linear_1 = nn.Linear(in_channel, FFN_channel)
        self.conv1 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.linear_2 = nn.Linear(FFN_channel, out_channel)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear_1(x.permute(0, 2, 3, 1).reshape(B, -1, C)).permute(0, 2, 1).reshape(B, self.FFN_channel, H, W)
        rs2 = self.act(self.conv1(rs1))
        rs3 = self.conv2(rs2) + rs1
        rs4 = self.linear_2(rs3.permute(0, 2, 3, 1).reshape(B, -1, self.FFN_channel)).permute(0, 2, 1).reshape(B, self.out_channel, H, W)
        return rs4


class FFN_2(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(FFN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, FFN_channel, 3, 1, 2, bias=True, dilation=2)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(FFN_channel, out_channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)

    def forward(self, x):
        rs1 = self.conv1(x)
        rs2 = self.act(self.conv2(rs1))
        rs3 = self.conv3(rs2) + rs1
        rs4 = self.conv4(rs3)
        return rs4


class conv_IDWT(nn.Module):
    def __init__(self, channel):
        super(conv_IDWT, self).__init__()
        self.res_block = resblock(channel=channel)
        self.IDWT = IDWT_2D()

    def forward(self, x):
        rs1 = self.IDWT(x)
        rs2 = self.res_block(rs1)
        return rs2


class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)

    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2


class DWC(nn.Module):
    def __init__(self, channel):
        super(DWC, self).__init__()
        self.linear = nn.Linear(channel, channel, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear(x.permute(0, 2, 3, 1).reshape(B, -1, C))
        rs2 = self.sigmoid(rs1).permute(0, 2, 1).reshape(B, C, H, W)
        return rs2


class Attention(nn.Module):
    def __init__(self, channel, head_channel, dropout):
        super(Attention, self).__init__()
        self.head_channel, self.channel = head_channel, channel
        self.q = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.k = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.scale = head_channel ** 0.5
        self.num_head = channel // self.head_channel
        self.mlp_1 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, q, k, v):
        B, q_C, H, W = q.shape
        _, v_C, _, _ = v.shape
        q_attn = self.q(q.permute(0, 2, 3, 1).reshape(B, -1, q_C)).reshape(B, -1, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        k_attn = self.k(k.permute(0, 2, 3, 1).reshape(B, -1, q_C)).reshape(B, -1, self.num_head, self.head_channel).permute(0, 2, 3, 1)
        v_attn_1 = self.v(v.permute(0, 2, 3, 1).reshape(B, -1, v_C))
        v_attn = v_attn_1.reshape(B, -1, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        attn = ((q_attn @ k_attn) / self.scale).softmax(dim=-1)
        x = (attn @ v_attn).permute(0, 2, 1, 3).reshape(B, -1, v_C)
        rs1 = v_attn_1.permute(0, 2, 1).reshape(B, q_C, H, W) + self.mlp_1(x).permute(0, 2, 1).reshape(B, v_C, H, W)
        rs2 = rs1 + self.mlp_2(rs1.permute(0, 2, 3, 1).reshape(B, -1, v_C)).permute(0, 2, 1).reshape(B, v_C, H, W)
        return rs2


class combine(nn.Module):
    def __init__(self, channel):
        super(combine, self).__init__()
        self.resblock = resblock(channel=channel)
        self.a = nn.Parameter(torch.tensor(0.33), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.33), requires_grad=True)

    def forward(self, x1, x2, x3):
        rs1 = self.a * x1 + self.b * x2 + (1 - self.a - self.b) * x3
        rs2 = self.resblock(rs1)
        return rs2


class S_MWiT(nn.Module):
    def __init__(self, pan_ll_channel, L_up_channel, head_channel, dropout):
        super(S_MWiT, self).__init__()
        self.pan_ll_channel = pan_ll_channel
        self.WD = DWT_2D()
        self.v_ll_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_lh_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hl_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hh_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.conv_idwt_pan = conv_IDWT(channel=pan_ll_channel)
        self.wd_ll_conv = DWC(channel=pan_ll_channel)
        self.wd_lh_conv = DWC(channel=pan_ll_channel)
        self.wd_hl_conv = DWC(channel=pan_ll_channel)
        self.wd_hh_conv = DWC(channel=pan_ll_channel)
        self.conv_idwt_up = conv_IDWT(channel=L_up_channel)
        self.combine = combine(channel=L_up_channel)
        self.resblock = resblock(channel=L_up_channel)
        self.resblock_1 = resblock(channel=L_up_channel)
        self.mlp = FFN(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_x = FFN_2(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_v = FFN_2(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)

    def forward(self, pan_ll, L_up, back_img):
        wd_ll, wd_lh, wd_hl, wd_hh = torch.split(self.WD(pan_ll), [self.pan_ll_channel, self.pan_ll_channel, self.pan_ll_channel, self.pan_ll_channel], dim=1)

        pre_v = self.combine(x1=wd_ll, x2=L_up, x3=self.mlp(back_img))
        v = self.resblock(pre_v)

        v_ll = self.v_ll_attn(q=wd_ll, k=wd_ll, v=v)
        v_lh = self.v_lh_attn(q=wd_lh, k=wd_ll, v=v)
        v_hl = self.v_hl_attn(q=wd_hl, k=wd_ll, v=v)
        v_hh = self.v_hh_attn(q=wd_hh, k=wd_ll, v=v)
        v_idwt = self.conv_idwt_up(torch.cat([v_ll, v_lh, v_hl, v_hh], dim=1))

        x_idwt = self.conv_idwt_pan(
            torch.cat([self.wd_ll_conv(wd_ll), self.wd_lh_conv(wd_lh), self.wd_hl_conv(wd_hl), self.wd_hh_conv(wd_hh)],
                      dim=1))
        x_1 = self.conv_x(x_idwt) + self.conv_v(v_idwt)
        x = self.resblock_1(x_1)
        return x


class F_MWiT(nn.Module):
    def __init__(self, pan_channel, L_up_channel, head_channel, dropout):
        super(F_MWiT, self).__init__()
        self.s_mwit = S_MWiT(pan_ll_channel=pan_channel, L_up_channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.combine = combine(channel=L_up_channel)
        self.mlp = FFN(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.resblock = resblock(channel=L_up_channel)

    def forward(self, pan, L_up, back_img, lms):
        x = self.s_mwit(pan_ll=pan, L_up=L_up, back_img=back_img)
        x = self.combine(x1=pan, x2=lms, x3=self.mlp(x))
        x = self.resblock(x)
        return x

# small scale
class L_MWiT(nn.Module):
    def __init__(self, pan_ll_channel, L_up_channel, head_channel, dropout):
        super(L_MWiT, self).__init__()
        self.pan_ll_channel = pan_ll_channel
        self.WD = DWT_2D()
        self.v_ll_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_lh_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hl_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hh_attn = Attention(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.mlp = FFN(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.wd_ll_conv = DWC(channel=pan_ll_channel)
        self.wd_lh_conv = DWC(channel=pan_ll_channel)
        self.wd_hl_conv = DWC(channel=pan_ll_channel)
        self.wd_hh_conv = DWC(channel=pan_ll_channel)
        self.conv_idwt_pan = conv_IDWT(channel=pan_ll_channel)
        self.conv_idwt_up = conv_IDWT(channel=L_up_channel)
        self.combine = combine(channel=L_up_channel)
        self.resblock = resblock(channel=L_up_channel)
        self.resblock_1 = resblock(channel=L_up_channel)
        self.conv_x = FFN_2(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_v = FFN_2(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)

    def forward(self, pan_ll, back_img, L_up):
        wd_ll, wd_lh, wd_hl, wd_hh = torch.split(self.WD(pan_ll),
                                                 [self.pan_ll_channel, self.pan_ll_channel, self.pan_ll_channel,
                                                  self.pan_ll_channel], dim=1)

        pre_v = self.combine(x1=wd_ll, x2=L_up, x3=self.mlp(back_img))
        v = self.resblock(pre_v)

        v_ll = self.v_ll_attn(q=wd_ll, k=wd_ll, v=v)
        v_lh = self.v_lh_attn(q=wd_lh, k=wd_ll, v=v)
        v_hl = self.v_hl_attn(q=wd_hl, k=wd_ll, v=v)
        v_hh = self.v_hh_attn(q=wd_hh, k=wd_ll, v=v)
        v_idwt = self.conv_idwt_up(torch.cat([v_ll, v_lh, v_hl, v_hh], dim=1))

        x_idwt = self.conv_idwt_pan(
            torch.cat([self.wd_ll_conv(wd_ll), self.wd_lh_conv(wd_lh), self.wd_hl_conv(wd_hl), self.wd_hh_conv(wd_hh)],
                      dim=1))
        x_1 = self.conv_x(x_idwt) + self.conv_v(v_idwt)
        x = self.resblock_1(x_1)
        return x


class HWViT(nn.Module):
    def __init__(self, L_up_channel, pan_channel, pan_target_channel, ms_target_channel, head_channel, dropout):
        super(HWViT, self).__init__()
        self.pan_channel = pan_channel
        self.lms = nn.Sequential(
            nn.Conv2d(L_up_channel, L_up_channel * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        self.pan_raise_channel = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
        self.lms_raise_channel = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.ms_raise_channel = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.reduce_channel = reduce_channel(ms_target_channel=ms_target_channel, L_up_channel=L_up_channel)
        self.F_MWiT_block = F_MWiT(L_up_channel=ms_target_channel, pan_channel=pan_target_channel, head_channel=head_channel, dropout=dropout)
        self.L_MWiT_block = L_MWiT(L_up_channel=ms_target_channel, pan_ll_channel=pan_target_channel, head_channel=head_channel, dropout=dropout)
        self.lms_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lms_down_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pan_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act_1 = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.act_2 = nn.PReLU(num_parameters=L_up_channel, init=0.01)

    def forward(self, pan, ms, lms):
        pan = self.pan_raise_channel(pan)
        lms_1 = self.act_1(self.lms(ms) + lms)
        lms_2 = self.lms_raise_channel(lms_1)
        back_1 = self.L_MWiT_block(pan_ll=self.pan_down_2(pan), L_up=self.lms_down_4(lms_2), back_img=self.ms_raise_channel(ms))
        back_2 = self.F_MWiT_block(pan=pan, L_up=self.lms_down_2(lms_2), back_img=back_1, lms=lms_2)
        back = self.reduce_channel(back_2)
        result = self.act_2(back + lms_1)
        return result


