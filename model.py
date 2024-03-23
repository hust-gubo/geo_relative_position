class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_chans=3, embed_dim=384, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class Attention(nn.Module):
    def __init__(self, dim = 384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x, n_patches):
        B, N, C = x.shape
        #B N C -> B N 3C -> B N 3 H c -> 3 B H N c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q,k,v B H N c
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_cls = attn[:,:,0,2:n_patches + 2]
        attn_dist = attn[:,:,1,2:n_patches + 2]
        attn_cls = reduce(attn_cls, 'b h n -> b n',reduction='mean')
        attn_dist = reduce(attn_dist, 'b h n -> b n',reduction='mean')
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_cls, attn_dist


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536,bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(1536, 384,bias=True)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self,model_type):
        super(Block, self).__init__()
        self.norm1 = LayerNorm(384, eps=1e-6)
        self.norm2 = LayerNorm(384, eps=1e-6)
        self.mlp = Mlp(0.)
        self.attn = Attention()
        self.model_type = model_type
        if model_type == 'sat':
            self.maxpool = torch.nn.MaxPool2d((2,2),(2,2), return_indices=True)
        else:
            self.maxpool1 = torch.nn.MaxPool2d((2,3),(2,3), return_indices=True)
            self.maxpool2 = torch.nn.MaxPool2d((2,1),(2,1), return_indices=True)
            self.maxpool3 = torch.nn.MaxPool2d((3,1),(3,1), return_indices=True)
        self.attention_pool_cls = Linear(384,1)
        self.attention_pool_dist = Linear(384,1)
        self.mlp_trans = Mlp(0.)

    def get_relative_sat(self, attn, x):

        att = rearrange(attn, 'b (h w) -> b h w', h = 16)
        attcenter = att[:,4:12,4:12]
        attcenter = rearrange(attcenter, 'b h w -> b (h w)')
        _, indexc = torch.topk(attcenter,48)
        indexc = indexc.unsqueeze(-1).expand(-1,-1,384)

        _, indexr = self.maxpool(att)
        indexr = rearrange(indexr, 'b h w -> b (h w)').unsqueeze(-1).expand(-1, -1, 384)
        
        add_token = x
        add_token = rearrange(add_token,'b (h w) c -> b h w c',h = 16)
        add_token_center = add_token[:,4:12,4:12,:]
        add_token_center = rearrange(add_token_center, 'b h w c -> b (h w) c')
        add_token_roud = rearrange(add_token,'b h w c -> b (h w) c')
        # print(add_token_roud.size())
        # print(indexr.size())
        add_token_roud = torch.gather(add_token_roud, 1, indexr)
        add_token_center = torch.gather(add_token_center, 1, indexc)
        add_token = torch.cat([add_token_center,add_token_roud],dim = 1)
        return add_token

    def get_relative_grd(self, attn, x):

        attn = rearrange(attn, 'b (h w) -> b h w', h = 7)
        atttop = attn[:,0:2,:]
        attmid = attn[:,2:4,:]
        attbot = attn[:,4:7,:]
        attmax, indextop = self.maxpool1(atttop)
        attmax, indexmid = self.maxpool2(attmid)
        attmax, indexbot = self.maxpool3(attbot)
        indextop = rearrange(indextop, 'b h w -> b (h w)')
        indextop = indextop.unsqueeze(-1).expand(-1, -1, 384) 
        indexbot = rearrange(indexbot, 'b h w -> b (h w)')
        indexbot = indexbot.unsqueeze(-1).expand(-1, -1, 384) 
        indexmid = rearrange(indexmid, 'b h w -> b (h w)')
        indexmid = indexmid.unsqueeze(-1).expand(-1, -1, 384) 
        add_token = x
        add_token = rearrange(add_token,'b (h w) c ->b h w c', h =7)
        add_token_top = rearrange(add_token[:,0:2,:,:],'b h w c -> b (h w) c')
        add_token_mid = rearrange(add_token[:,2:4,:,:],'b h w c -> b (h w) c')
        add_token_bot = rearrange(add_token[:,4:7,:,:],'b h w c -> b (h w) c')


        add_token_top = torch.gather(add_token_top, 1, indextop)
        add_token_bot = torch.gather(add_token_bot, 1, indexbot)
        add_token_mid = torch.gather(add_token_mid, 1, indexmid)
        add_token = torch.cat([add_token_top,add_token_mid, add_token_bot],dim = 1)
        return add_token
    def forward(self, x, n_patches):
        h = x
        x = self.norm1(x)
        x, attn_cls, attn_dist = self.attn(x, n_patches)
        B, N, C = x.shape
        if self.model_type == 'sat':
            add_token = x[:,2:n_patches + 2,:]
            add_token_cls = self.get_relative_sat(attn_cls, add_token)
            add_token_cls = torch.matmul(F.softmax(self.attention_pool_cls(add_token_cls), dim=1).transpose(-1, -2), add_token_cls)
            # add_token_dist = self.get_relative_sat(attn_dist, add_token)
            # add_token_dist = torch.matmul(F.softmax(self.attention_pool_dist(add_token_dist), dim=1).transpose(-1, -2), add_token_dist)
            # add_token_trans_label = torch.cat([add_token_cls, add_token_dist], dim=1)
            add_token_trans = self.mlp_trans(add_token_cls)
        else:
            add_token = x[:,2:n_patches + 2,:]
            add_token_cls = self.get_relative_grd(attn_cls, add_token)
            add_token_cls = torch.matmul(F.softmax(self.attention_pool_cls(add_token_cls), dim=1).transpose(-1, -2), add_token_cls)
            # add_token_dist = self.get_relative_grd(attn_dist, add_token)
            # add_token_dist = torch.matmul(F.softmax(self.attention_pool_dist(add_token_dist), dim=1).transpose(-1, -2), add_token_dist)
            # add_token_trans_label = torch.cat([add_token_cls, add_token_dist], dim=1)
            add_token_trans = self.mlp_trans(add_token_cls)
        # attmax, index = self.maxpool(att)
        # index = rearrange(index, 'b h w -> b (h w)')
        # index = index.unsqueeze(-1).expand(-1, -1, 768) 
        # add_token = x[:,1:257,:]
        # add_token = torch.gather(add_token, 1, index)
        x = x + h
        x = torch.cat([x,add_token_cls,add_token_dist,add_token_trans],dim = 1)


        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x, add_token_trans_label, add_token_trans

class Deit(nn.Module):
    def __init__(self,img_size):
        super(Deit, self).__init__()
        self.dist_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        # print(img_size[0] // 16)
        # print(img_size[1] // 16)
        n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        self.n_patches = n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+2, 384))
        self.patch_embed = PatchEmbed(img_size)
        self.blocks = nn.ModuleList()
        if img_size[0] == img_size[1]:
            model_type = 'sat'
        else:
            model_type = 'grd'
        for _ in range(12):
            layer = Block(model_type)
            self.blocks.append(layer)
        self.norm = LayerNorm(384, eps=1e-6)
        self.head = Linear(in_features=384, out_features=1000, bias=True)
        self.head_dist = Linear(in_features=384, out_features=1000, bias=True)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        add_token = None
        add_token_label = None
        for i, blk in enumerate(self.blocks):
            x, add_token_trans_label, add_token_trans = blk(x, self.n_patches)
            if add_token == None:
                add_token = add_token_trans
                add_token_label = add_token_trans_label
            else:
                add_token = torch.cat([add_token, add_token_trans],dim = 1)
                add_token_label = torch.cat([add_token_label, add_token_trans_label],dim = 1)
        x = self.norm(x)
        x_dist = x[:, 1]
        x = x[:, 0]
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2 , add_token_label, add_token

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.query_net = Deit(img_size=(112,616))
        self.reference_net = Deit(img_size=(256,256))
        self.set_pretrain_all()
    def forward(self, im_q, im_k):
        grd_out, grd_label, grd_trans = self.query_net(im_q)
        sat_out, sat_label, sat_trans = self.reference_net(x=im_k)
        return grd_out, sat_out, grd_label, grd_trans, sat_label, sat_trans

    def set_pretrain(self, img_size):
        # checkpoint = torch.load(path,map_location=torch.device('cpu'))
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // 16, img_size[1] // 16)
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        for i in range(12):
            keys = "blocks."+str(i)+".mlp.fc1.weight"
            checkpoint['model']["blocks."+str(i)+".mlp_trans.fc1.weight"] = checkpoint['model'][keys]
            keys = "blocks."+str(i)+".mlp.fc1.bias"
            checkpoint['model']["blocks."+str(i)+".mlp_trans.fc1.bias"] = checkpoint['model'][keys]
            keys = "blocks."+str(i)+".mlp.fc2.weight"
            checkpoint['model']["blocks."+str(i)+".mlp_trans.fc2.weight"] = checkpoint['model'][keys]
            keys = "blocks."+str(i)+".mlp.fc2.bias"
            checkpoint['model']["blocks."+str(i)+".mlp_trans.fc2.bias"] = checkpoint['model'][keys]
        return checkpoint["model"]

    def set_pretrain_all(self):
        logger.info("load...")
        self.query_net.load_state_dict(self.set_pretrain((112,616)),strict = False)
        self.reference_net.load_state_dict(self.set_pretrain((256,256)),strict = False)
        logger.info("load complete!")

