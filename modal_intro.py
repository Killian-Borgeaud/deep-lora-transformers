import modal

app = modal.App("modal-intro")

image = modal.Image.debian_slim().uv_sync().add_local_dir("dlt", remote_path="/root/dlt").add_local_dir("scripts", remote_path="/root/scripts")

vol = modal.Volume.from_name("dlt-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10",
    timeout=60*60,
    volumes={"/root/checkpoints": vol},
    )
def run_intro():
    from scripts.intro import main
    main()
    vol.commit()


@app.local_entrypoint()
def main():
    result = run_intro.remote()
    print(result)