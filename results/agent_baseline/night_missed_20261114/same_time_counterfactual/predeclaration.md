# 固定六例时序不变性反事实

固定UID来自已有review_pack/review.md，不看本实验回答选例：second_appeared:train_1065:1、:4；second_disappeared:train_1065:1、:2；levir_any:train_106:1、:3。每例分别T1/T1与T2/T2，共12配置，串行qwen3.7-flash，每配置最多一次请求，失败不自动重试。已有成功记录恢复时跳过；attempted但无成功记录停止以避免重复请求。

使用原始RGB与原mask/bbox重新调用ask_images(context=True)。该函数顺序是T1近景、T2近景、T1宽景、T2宽景；保留原文字标题，替换全部四图，不复制旧时相面板。渲染前检查两期RGB逐像素相同。保留原V0完整文字、方向/scope和请求参数（包括seed）；不读GT。原V0仅作对照，不复用它作为新回答。

这是人为构造的无时相变化条件；yes违反时序不变性，可提示基本时序判断问题。它不是自然任务收益、人工语义像素真值或线上晋级依据。六例已有视觉曝光，不是盲选独立测试。12配置后结束该分支，不继续提示词/阈值扫描。

保存design、每次实际四图/request、raw响应/HTTP状态/error_body/usage计量、原V0与反事实对照。HTTP状态如底层成功事件未提供则保留null，不臆造状态；usage折算费用不是账户账单。主Agent审核和Slurm提交，此记录本身不代表已执行。
