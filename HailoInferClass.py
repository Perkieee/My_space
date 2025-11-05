from typing import Tuple, Dict, List
from typing import Callable, Optional
from functools import partial
import numpy as np

from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from hailo_platform.pyhailort.pyhailort import FormatOrder



class HailoInfer:
    def __init__(
        self, hef_path: str, batch_size: int = 1,
            input_type: Optional[str] = None, output_type: Optional[str] = None,
            priority: Optional[int] = 0) -> None:

        """
        Initialize the HailoAsyncInference class to perform asynchronous inference using a Hailo HEF model.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (optional[int]): Number of inputs processed per inference. Defaults to 1.
            input_type (Optional[str], optional): Input data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
            output_type (Optional[str], optional): Output data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
            priority (optional[int]): Scheduler priority value for the model within the shared VDevice context. Defaults to 0.
        """

        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        vDevice = VDevice(params)

        self.target = vDevice
        self.hef = HEF(hef_path)

        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        self._set_input_type(input_type)
        self._set_output_type(output_type)

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(priority)
        self.last_infer_job = None


    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """

        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

    def _set_output_type(self, output_type: Optional[str] = None) -> None:
        """
        Set the output type for each model output.

        Args:
            output_type (Optional[str]): Desired output data type. Common values:
                'UINT8', 'UINT16', 'FLOAT32'.
        """

        self.nms_postprocess_enabled = False

        # If the model uses HAILO_NMS_WITH_BYTE_MASK format (e.g.,instance segmentation),
        if self.infer_model.outputs[0].format.order == FormatOrder.HAILO_NMS_WITH_BYTE_MASK:
            # Use UINT8 and skip setting output formats
            self.nms_postprocess_enabled = True
            self.output_type = self._output_data_type2dict("UINT8")
            return

        # Otherwise, set the format type based on the provided output_type argument
        self.output_type = self._output_data_type2dict(output_type)

        # Apply format to each output layer
        for name, dtype in self.output_type.items():
            self.infer_model.output(name).set_format_type(getattr(FormatType, dtype))


    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get a HEF instance
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input


    def run_async(self, input_batch: List[np.ndarray], inference_callback_fn) -> object:
        """
        Run an asynchronous inference job on a batch of preprocessed inputs.

        This method reuses a preconfigured model (no reconfiguration overhead),
        prepares input/output bindings, launches async inference, and returns
        the job handle so that the caller can wait on it if needed.

        Args:
            input_batch (List[np.ndarray]): A batch of preprocessed model inputs.
            inference_callback_fn (Callable): Function to be invoked when inference is complete.
                                              It receives `bindings_list` and additional context.

        Returns:
            None
        """
        bindings_list = self.create_bindings(self.configured_model, input_batch)
        self.configured_model.wait_for_async_ready(timeout_ms=10000)

        # Launch async inference and attach the result handler
        self.last_infer_job = self.configured_model.run_async(
            bindings_list,
            partial(inference_callback_fn, bindings_list=bindings_list)
        )
    
    def _extract_outputs(self, bindings_list):
        """
        Extracts and post-processes model outputs from synchronous inference results.
        This method is designed for classification models.
    
        Args:
            bindings_list (list[Dict]): A list of inference result bindings.
                Each binding should contain the output buffers as returned by
                `InferVStreams.infer()` or `self.configured_model.run()`.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries containing:
                - 'raw_output': raw output tensor (NumPy array)
                - 'probabilities': normalized class probabilities
                - 'top1_class': class index with highest probability
                - 'top1_score': top-1 confidence score (float)
                - 'top5': list of (class_idx, probability) tuples for top-5 classes
        """

        results = []

        # Go through each set of inference results (one per batch element)
        for binding in bindings_list:
            # --- Step 1: Retrieve raw output tensor ---
            # Depending on SDK version, output() may return dict or ndarray
            output_buffers = binding.output().get_buffer() if hasattr(binding, "output") else binding
            if isinstance(output_buffers, dict):
                # Take the first output tensor (assuming single-head classifier)
                output = list(output_buffers.values())[0]
            else:
                output = output_buffers

            # Ensure output is a NumPy array
            output = np.array(output).astype(np.float32)

            # Flatten to 1D for classification head
            output_flat = output.flatten()

            # --- Step 2: Convert logits to probabilities ---
            # Apply softmax normalization
            exp_scores = np.exp(output_flat - np.max(output_flat))
            probs = exp_scores / np.sum(exp_scores)

            # --- Step 3: Extract top predictions ---
            top5_idx = np.argsort(probs)[-5:][::-1]
            top1_idx = int(top5_idx[0])
            top1_score = float(probs[top1_idx])

            # Optionally decode class labels if available
            label = None
            if hasattr(self, "labels") and isinstance(self.labels, dict):
                label = self.labels.get(top1_idx, f"Class {top1_idx}")
            else:
                label = f"Class {top1_idx}"

            # --- Step 4: Structure output ---
            result = {
                "raw_output": output_flat,
                "probabilities": probs,
                "top1_class": top1_idx,
                "top1_label": label,
                "top1_score": top1_score,
                "top5": [(int(i), float(probs[i])) for i in top5_idx],
            }

            results.append(result)

        return results


    def run_sync(self, input_batch: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run synchronous inference (blocks until completion).

        Args:
            input_batch (List[np.ndarray]): Preprocessed model inputs.

        Returns:
            Dict[str, np.ndarray]: Output tensors for this inference batch.
        """
        # Prepare I/O bindings
        bindings_list = self.create_bindings(self.configured_model, input_batch)
    
        # Run inference (blocking call)
        self.configured_model.run(bindings_list,timeout_ms=10000)
    
        # Collect and return results
        outputs = self._extract_outputs(bindings_list)
        return outputs

    def create_bindings(self, configured_model, input_batch):
        """
        Create a list of input-output bindings for a batch of frames.

        Args:
            configured_model: The configured inference model.
            input_batch (List[np.ndarray]): List of input frames, preprocessed and ready.

        Returns:
            List[Bindings]: A list of bindings for each frame's input and output buffers.
        """

        def frame_binding(frame: np.ndarray):
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
                for name in self.output_type
            }

            binding = configured_model.create_bindings(output_buffers=output_buffers)
            binding.input().set_buffer(np.array(frame))
            return binding

        return [frame_binding(frame) for frame in input_batch]



    def is_nms_postprocess_enabled(self) -> bool:
        """
        Returns True if the HEF model includes an NMS postprocess node.
        """
        return self.nms_postprocess_enabled

    def _output_data_type2dict(self, data_type: Optional[str]) -> Dict[str, str]:
        """
        Generate a dictionary mapping each output layer name to its corresponding
        data type. If no data type is provided, use the type defined in the HEF.

        Args:
            data_type (Optional[str]): The desired data type for all output layers.
                                       Valid values: 'float32', 'uint8', 'uint16'.
                                       If None, uses types from the HEF metadata.

        Returns:
            Dict[str, str]: A dictionary mapping output layer names to data types.
        """
        valid_types = {"float32", "uint8", "uint16"}
        data_type_dict = {}

        for output_info in self.hef.get_output_vstream_infos():
            name = output_info.name
            if data_type is None:
                # Extract type from HEF metadata
                hef_type = str(output_info.format.type).split(".")[-1]
                data_type_dict[name] = hef_type
            else:
                if data_type.lower() not in valid_types:
                    raise ValueError(f"Invalid data_type: {data_type}. Must be one of {valid_types}")
                data_type_dict[name] = data_type

        return data_type_dict


    def close(self):

        # Wait for the final job to complete before exiting
        if self.last_infer_job is not None:
            self.last_infer_job.wait(10000)

        if self.config_ctx:
            self.config_ctx.__exit__(None, None, None)
