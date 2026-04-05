import torch


def save_checkpoint(
    path: str,
    world_model,
    actor,
    critic,
    world_model_optimizer,
    actor_optimizer,
    critic_optimizer,
    step: int,
    gradient_step: int,
):
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "world_model_optimizer": world_model_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "step": step,
            "gradient_step": gradient_step,
        },
        path,
    )


def load_checkpoint(
    path: str,
    world_model,
    actor,
    critic,
    world_model_optimizer,
    actor_optimizer,
    critic_optimizer,
    device: str,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    world_model.load_state_dict(checkpoint["world_model"])
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return {
        "step": checkpoint["step"],
        "gradient_step": checkpoint["gradient_step"],
    }
