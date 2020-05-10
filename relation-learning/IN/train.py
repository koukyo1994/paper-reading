import models
import utils

from catalyst.dl import SupervisedRunner

if __name__ == "__main__":
    args = utils.get_parser().parse_args()

    config = utils.load_config(args.config)

    global_params = config["globals"]

    utils.set_seed(global_params["seed"])
    device = utils.get_device(global_params)
    output_dir = global_params["output_dir"]

    data_conf = config["data"]
    if args.generate:
        for c in data_conf.values():
            utils.generate_data(c)

    model = models.get_model(config).to()
    criterion = utils.get_criterion(config)
    optimizer = utils.get_optimizer(model, config)
    scheduler = utils.get_scheduler(optimizer, config)
    loaders = {
        phase: utils.get_loader(config, phase)
        for phase in ["train", "valid3", "valid", "valid12"]
    }

    runner = SupervisedRunner(
        device=device,
        input_key=["objects", "externals", "triplet"],
        input_target_key="targets")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        scheduler=scheduler,
        num_epochs=global_params["num_epochs"],
        verbose=True,
        logdir=output_dir)
