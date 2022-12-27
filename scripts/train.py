
import json
import numpy as np
from os.path import join
import diffuser.utils as utils
from diffuser.guides.policies import Policy
from torch.utils.tensorboard import SummaryWriter
import pdb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim * args.jump


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

eval_args = Parser().parse_args('plan')
eval_writer = SummaryWriter(log_dir=args.savepath)
def evaluate(epoch, sample_idx):


    env_eval = dataset.load_environment(eval_args.dataset)
    policy = Policy(trainer.ema_model, dataset.normalizer)

    observation = env_eval.reset()
    target = env_eval._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }
    observation = env_eval.reset()
    init_state = env_eval.state_vector()

    for infer_action in [True, False]:

        ## observations for rendering
        env_eval.set_state(init_state[:2], init_state[2:])
        observation = env_eval._get_obs()
        rollout = [observation.copy()]

        total_reward = 0
        for t in range(env_eval.max_episode_steps // args.jump):

            state = env_eval.state_vector().copy()

            ## can replan if desired, but the open-loop plans are good enough for maze2d
            ## that we really only need to plan once
            if t == 0:
                cond[0] = observation

                action, samples = policy(cond, batch_size=eval_args.batch_size)
                actions = samples.actions[0]
                actions = actions.reshape(diffusion.horizon, args.jump, -1)
                sequence = samples.observations[0]
            # pdb.set_trace()

            # ####
            if t < len(sequence) - 1:
                next_waypoint = sequence[t+1]
            else:
                next_waypoint = sequence[-1].copy()
                next_waypoint[2:] = 0
                # pdb.set_trace()

            ## can use actions or define a simple controller based on state predictions
            action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            for j in range(args.jump):

                if infer_action:
                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                else:
                    if t < len(sequence) - 1:
                        action = actions[t, j]
                    else:
                        next_waypoint = sequence[-1].copy()
                        next_waypoint[2:] = 0
                        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

                next_observation, reward, terminal, _ = env_eval.step(action)
                total_reward += reward
                score = env_eval.get_normalized_score(total_reward)
                print(
                    f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                    f'{action}'
                )

                if 'maze2d' in eval_args.dataset:
                    xy = next_observation[:2]
                    goal = env_eval.unwrapped._target
                    print(
                        f'maze | pos: {xy} | goal: {goal}'
                    )

                ## update rollout observations
                rollout.append(next_observation.copy())

            # logger.log(score=score, step=t)

            if t % eval_args.vis_freq == 0 or terminal:
                fullpath = join(eval_args.savepath, f'J{args.jump}-inferAct{infer_action}-e{epoch}-s{sample_idx}-{t}.png')


                if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


                # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                ## save rollout thus far
                renderer.composite(join(eval_args.savepath, f'rollout_J{args.jump}-inferAct{infer_action}_e{epoch}-s{sample_idx}.png'), np.array(rollout)[None], ncol=1)

                # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

                # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

            if terminal:
                break

            observation = next_observation

        # logger.finish(t, env.max_episode_steps, score=score, value=0)

        eval_writer.add_scalars(f'rollout_J{args.jump}-inferAct{infer_action}', 
                                {'score': score, 'return': total_reward}, global_step=epoch*args.n_steps_per_epoch+sample_idx)
        ## save result as a json file
        json_path = join(args.savepath, 'rollout.json')
        json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': epoch}
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)



n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
eval_sample_n = 5

for i in range(n_epochs):

    print(f'Epoch {i} / {n_epochs} | {args.savepath}')

    for k in range(eval_sample_n):
        evaluate(i, 0)

    trainer.train(n_train_steps=args.n_steps_per_epoch)

