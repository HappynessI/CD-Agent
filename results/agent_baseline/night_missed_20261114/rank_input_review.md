# 排名核验输入复核

代码与实际模板只读审查：`rank_V0/prepared/second_appeared/train_1065.json`，模板UID `second_appeared:train_1065:1`，原request来自smoke四图。文本仅包含building scope、appeared方向与红轮廓内对象的通用指代，无旧候选坐标、面积、特定对象描述。

V0新增候选调用原ask_images(context=True)，四图均替换、JPEG quality=92；旧答案仅在完整mask精确一致时从four分支引用，保留reused_from与原费用，新费用为0。新请求模型仍qwen3.7-flash，seed按新UID设置；这是缓存复用配对，不是新的独立seed重复实验。没有使用Spotlight renderer冒充原V0。

本检查未修改在跑作业。186463在准备23/30时Slurm记录CPU约46分、RSS约1.5GB，错误日志仅显示8项既有检查通过，无API错误；处于计算阶段，不能将耗时解释为连通性失败。
