import torch
import torch.nn as nn
import torch.nn.functional as F

config={
    'in_channel': 6,
    'hidden_dim': 32,
    'encoder_kernel_size': 3,
    'encoder_stride': 2,
    'triple_pass_filter': 256
}


class PaddedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, ks, stride):
        super().__init__()
        # Custom Padding Calculation
        if isinstance(ks, tuple):
            k_h, k_w = ks
        else:
            k_h = ks
            k_w = ks
        if isinstance(stride, tuple):
            s_h, s_w = stride
        else:
            s_h = stride
            s_w = stride
        pad_h, pad_w = k_h - s_h, k_w - s_w
        pad_up, pad_left = pad_h // 2, pad_w // 2
        pad_down, pad_right= pad_h - pad_up, pad_w - pad_left
        self.pad = nn.ReflectionPad2d([pad_left, pad_right, pad_up, pad_down])
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=ks, stride=stride, bias=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class NHDRRNet(nn.Module):
    def __init__(self) -> None:
        super(NHDRRNet, self).__init__()
        self.filter = config.filter
        self.encoder_kernel = config.encoder_kernel
        self.decoder_kernel = config.decoder_kernel
        self.triple_pass_filter = config.triple_pass_filter
        self.c_dim = 3

        self.encoder_1 = []
        self.encoder_2 = []
        self.encoder_3 = []
        self.encoder_1.append(self._make_encoder(config['in_channel'], config['hidden_dim']))
        self.encoder_1.append(self._make_encoder(config['hidden_dim'], config['hidden_dim'] * 2))
        self.encoder_1.append(self._make_encoder(config['hidden_dim'] * 2, config['hidden_dim'] * 4))
        self.encoder_1.append(self._make_encoder(config['hidden_dim'] * 4, config['hidden_dim'] * 8))

        self.encoder_2.append(self._make_encoder(config['in_channel'], config['hidden_dim']))
        self.encoder_2.append(self._make_encoder(config['hidden_dim'], config['hidden_dim'] * 2))
        self.encoder_2.append(self._make_encoder(config['hidden_dim'] * 2, config['hidden_dim'] * 4))
        self.encoder_2.append(self._make_encoder(config['hidden_dim'] * 4, config['hidden_dim'] * 8))

        self.encoder_3.append(self._make_encoder(config['in_channel'], config['hidden_dim']))
        self.encoder_3.append(self._make_encoder(config['hidden_dim'], config['hidden_dim'] * 2))
        self.encoder_3.append(self._make_encoder(config['hidden_dim'] * 2, config['hidden_dim'] * 4))
        self.encoder_3.append(self._make_encoder(config['hidden_dim'] * 4, config['hidden_dim'] * 8))
        self.final_encoder = nn.Sequential(
                                PaddedConv2d(config['hidden_dim'] * 8 * 3, config['triple_pass_filter'], 1, 1),
                                nn.BatchNorm2d(config['triple_pass_filter'], momentum=0.9),
                                nn.ReLU()
                            )
        self.triple_list = []
        for i in range(10):
            self.triple_list.append(self._make_triple_pass_layer())
        self.avgpool = nn.AdaptiveAvgPool2d(16, 16)
        self.theta_conv = PaddedConv2d(config['triple_pass_filter'], 128, 1, 1)
        self.phi_conv = PaddedConv2d(config['triple_pass_filter'], 128, 1, 1)
        self.g_conv = PaddedConv2d(config['triple_pass_filter'], 128, 1, 1)
        self.theta_phi_g_conv = PaddedConv2d(128, config['triple_pass_filter'], 1, 1)
        self.decoder1 = self._make_decoder(config['triple_pass_filter'] * 2, config['hidden_dim'] * 4)
        self.decoder2 = self._make_decoder(config['hidden_dim'] * 4, config['hidden_dim'] * 2)
        self.decoder3 = self._make_decoder(config['hidden_dim'] * 2, config['hidden_dim'])
        self.decoder_final = nn.Sequential(
            nn.ConvTranspose2d(config['hidden_dim'], 3, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
    def _make_encoder(self, in, out):
        encoder = nn.Sequential(
            PaddedConv2d(in, out, config['encoder_kernel_size'], config['encoder_stride']),
            nn.BatchNorm2d(out, momentum=0.9),
            nn.ReLU()
        )  
    # def _make_encoder(self):
    #     encoder = nn.Sequential(
    #                     PaddedConv2d(config['in_channel'], config['hidden_dim'], config['encoder_kernel_size'], config['encoder_stride']),
    #                     nn.BatchNorm2d(),
    #                     nn.ReLU(),
    #                     PaddedConv2d(config['hidden_dim'], config['hidden_dim'] * 2, config['encoder_kernel_size'], config['encoder_stride']),
    #                     nn.BatchNorm2d(),
    #                     nn.ReLU(),
    #                     PaddedConv2d(config['hidden_dim'] * 2, config['hidden_dim'] * 4, config['encoder_kernel_size'], config['encoder_stride']),
    #                     nn.BatchNorm2d(),
    #                     nn.ReLU(),
    #                     PaddedConv2d(config['hidden_dim'] * 4, config['hidden_dim'] * 8, config['encoder_kernel_size'], config['encoder_stride']),
    #                     nn.BatchNorm2d(),
    #                     nn.ReLU()
    #     )
    #     return encoder

    def _make_decoder(self, in, out):
        decoder = nn.Sequential(
            nn.ConvTranspose2d(in, out, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(out),
            nn.LeakyReLU()
        )
        return decoder

    def _make_triple_pass_layer(self):
        return [PaddedConv2d(config['triple_pass_filter'], config['triple_pass_filter'], 1, 1),
                PaddedConv2d(config['triple_pass_filter'], config['triple_pass_filter'], 3, 1),
                PaddedConv2d(config['triple_pass_filter'], config['triple_pass_filter'], 5, 1),
                PaddedConv2d(config['triple_pass_filter'], config['triple_pass_filter'], 3, 1)]
    
    def triplepass(self, x, triple):
        x1 = F.relu(triple[0](x))
        x2 = F.relu(triple[1](x))
        x3 = F.relu(triple[2](x))
        x3 = x1 + x2 + x3
        x4 = triple[3](x3)
        x5 = x4 + x

        return x5

    def global_non_local(self, x):
        b, _, h, w = x.shape
        theta = self.theta_conv(x).reshape(b, 128, h * w).permute(0, 2, 1)
        phi = self.phi_conv(x).reshape(b, 128, h * w)
        g = self.g_conv(x).reshape(b, 128, h * w).permute(0, 2, 1)

        theta_phi = F.softmax(torch.matmul(theta, phi))
        theta_phi_g = torch.matmul(theta_phi, g)
        theta_phi_g = theta_phi_g.permute(0, 2, 1).reshape(b, 128, h, w)

        theta_phi_g = self.theta_phi_g_conv(theta_phi_g)

        output = theta_phi_g + x

        return output

    def forward(self, in_LDR, in_HDR):
        image1 = torch.cat([in_LDR[:, 0:self.c_dim, :, :], in_HDR[:, 0:self.c_dim, :, :]], 1)
        image2 = torch.cat([in_LDR[:, self.c_dim:self.c_dim * 2, :, :], in_HDR[:, self.c_dim:self.c_dim * 2, :, :]], 1)
        image3 = torch.cat([in_LDR[:, self.c_dim * 2:self.c_dim * 3, :, :], in_HDR[:, self.c_dim * 2:self.c_dim * 3, :, :]], 1)

        x1_32 = self.encoder_1[0](image1)
        x1_64 = self.encoder_1[1](x1_32)
        x1_128 = self.encoder_1[2](x1_64)
        x1 = self.encoder_1[3](x1_128)

        x2_32 = self.encoder_2[0](image2)
        x2_64 = self.encoder_2[1](x2_32)
        x2_128 = self.encoder_2[2](x2_64)
        x2 = self.encoder_2[3](x2_128)

        x3_32 = self.encoder_3[0](image3)
        x3_64 = self.encoder_3[1](x3_32)
        x3_128 = self.encoder_3[2](x3_64)
        x3 = self.encoder_3[3](x3_128)

        x_cat = torch.cat([x1, x2, x3], dim=1)
        encoder_final = self.final_encoder(x_cat)

        for i in range(9):
            encoder_final = self.triplepass(encoder_final, self.triple_list[i])
        tpl_out = self.triplepass(encoder_final, self.triple_list[9])

        glb_out = self.avgpool(encoder_final)
        glb_out = self.global_non_local(glb_out)
        required_size = [encoder_final.shape[2], encoder_final.shape[3]]
        glb_out = F.interpolate(glb_out, size=required_size)

        out_512 = torch.cat([tpl_out, glb_out], dim=1)
        out_128 = self.decoder1(out_512)
        out_128 = out_128 + x1_128 + x2_128 + x3_128
        out_64 = self.decoder2(out_128)
        out_64 = out_64 + x1_64 + x2_64 + x3_64
        out_32 = self.decoder3(out_64)
        out_32 = out_32 + x1_32 + x2_32 + x3_32
        out =self.decoder_final(out_32)

        return out