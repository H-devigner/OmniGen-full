import tensorflow as tf

class OmniGenScheduler:
    """Scheduler for OmniGen model."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
    ):
        """Initialize scheduler."""
        if beta_schedule == "linear":
            betas = tf.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Glide paper
            betas = tf.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
            
        alphas = 1.0 - betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        alphas_cumprod_prev = tf.pad(alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)
        
        self.num_train_timesteps = num_train_timesteps
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.betas = betas
        self.alphas = alphas
        
        # Offset for deterministic sampling
        self.steps_offset = steps_offset
        
        # Clip sample for numerical stability
        self.clip_sample = clip_sample
        
        # Type of prediction
        self.prediction_type = prediction_type
        
        # For backwards compatibility
        self.set_alpha_to_one = set_alpha_to_one
        self.timestep_spacing = timestep_spacing
        
    def scale_model_input(self, sample: tf.Tensor, timestep: tf.Tensor) -> tf.Tensor:
        """Scale the denoising model input."""
        step_index = timestep
        sigma = tf.gather(self.sigmas, step_index)
        sigma = tf.cast(sigma, sample.dtype)
        
        # Expand sigma for proper broadcasting
        while len(sigma.shape) < len(sample.shape):
            sigma = tf.expand_dims(sigma, -1)
            
        scaled = sample / ((sigma**2 + 1) ** 0.5)
        return scaled
        
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference."""
        if self.timestep_spacing == "leading":
            timesteps = tf.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
            timesteps = tf.cast(timesteps, tf.int32)
        else:
            raise ValueError(f"Unsupported timestep spacing: {self.timestep_spacing}")
            
        sigmas = tf.sqrt(1 / self.alphas_cumprod - 1)
        sigmas = tf.concat([sigmas, tf.zeros([1], dtype=sigmas.dtype)], axis=0)
        
        self.timesteps = timesteps
        self.sigmas = sigmas
        self.num_inference_steps = num_inference_steps
        
    def step(
        self,
        model_output: tf.Tensor,
        timestep: tf.Tensor,
        sample: tf.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
    ):
        """Predict the sample at the previous timestep."""
        step_index = timestep
        sigma = tf.gather(self.sigmas, step_index)
        sigma = tf.cast(sigma, sample.dtype)
        
        # 1. compute predicted original sample from predicted noise also called "predicted x_0"
        if self.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Prediction type {self.prediction_type} not supported")
            
        # 2. clip predicted x_0
        if self.clip_sample:
            pred_original_sample = tf.clip_by_value(pred_original_sample, -1, 1)
            
        # 3. Get next step value
        step_index = step_index + 1
        sigma_next = tf.gather(self.sigmas, step_index)
        sigma_next = tf.cast(sigma_next, sample.dtype)
        
        # 4. compute variance
        sigma_up = tf.minimum(sigma_next, tf.sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2))
        sigma_up = tf.cast(sigma_up, sample.dtype)
        
        # 5. compute noise
        noise = tf.random.normal(
            sample.shape,
            dtype=sample.dtype,
            seed=generator.seed() if generator is not None else None
        )
        
        # 6. compute previous sample
        prev_sample = pred_original_sample + sigma_next * noise
        
        if not return_dict:
            return (prev_sample,)
            
        return {"prev_sample": prev_sample, "pred_original_sample": pred_original_sample}
