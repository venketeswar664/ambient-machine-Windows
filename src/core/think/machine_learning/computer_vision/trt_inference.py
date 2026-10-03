# # trt_inference.py

# import collections
# import torch
# import numpy as np
# import tensorrt as trt
# import contextlib
# import time


# class TimeProfiler(contextlib.ContextDecorator):
#     def __init__(self):
#         self.total = 0

#     def __enter__(self):
#         self.start = self.time()
#         return self

#     def __exit__(self, type, value, traceback):
#         self.total += self.time() - self.start

#     def reset(self):
#         self.total = 0

#     def time(self):
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         return time.time()


# class TRTInference:
#     """
#     Wrapper around a TensorRT engine that:
#       - Deserializes an engine from a file
#       - Allocates input/output bindings as Torch tensors
#       - Executes inference (single batch of size 1)
#       - Returns a dict of output tensors
#     """

#     def __init__(self,
#                  engine_path: str,
#                  device: str = "cuda:0",
#                  backend: str = "torch",
#                  max_batch_size: int = 32,
#                  verbose: bool = False,
#                  logger: trt.ILogger = None):
#         """
#         :param engine_path: Path to the serialized TensorRT .engine file
#         :param device: CUDA device string ("cuda:0") or "cpu"
#         :param backend: Only "torch" is supported in this implementation
#         :param max_batch_size: Maximum batch size to allocate for dynamic dimensions
#         :param verbose: If True, create a verbose TRT logger
#         :param logger: Optional Python logger to record messages
#         """
#         self.engine_path = engine_path
#         self.device = torch.device(device)
#         self.backend = backend
#         self.max_batch_size_for_bindings = max_batch_size
#         self.logger = logger

#         # Prepare a TensorRT logger
#         self.trt_logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

#         # TRT runtime/engine/context placeholders
#         self.runtime = None
#         self.engine = None
#         self.context = None

#         # These will be filled by get_bindings()
#         self.dynamic_input_shapes = {}
#         self.bindings = collections.OrderedDict()
#         self.bindings_addr = collections.OrderedDict()
#         self.input_names = []
#         self.output_names = []

#         # Attempt to build/deserializes the engine and create context
#         try:
#             if self.logger:
#                 self.logger.info(f"Initializing TensorRT runtime and deserializing engine from: {self.engine_path}")
#             trt.init_libnvinfer_plugins(self.trt_logger, "")

#             # Create TRT runtime
#             self.runtime = trt.Runtime(self.trt_logger)

#             # Deserialize engine
#             with open(self.engine_path, "rb") as f:
#                 engine_data = f.read()
#             self.engine = self.runtime.deserialize_cuda_engine(engine_data)
#             if self.engine is None:
#                 raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")

#             # Create execution context
#             self.context = self.engine.create_execution_context()
#             if self.context is None:
#                 raise RuntimeError("Failed to create TensorRT execution context")

#             # Build binding data structures (torch tensors) for each I/O
#             self.bindings = self._allocate_bindings(self.engine,
#                                                      self.context,
#                                                      self.max_batch_size_for_bindings,
#                                                      self.device)

#             # Build a map from binding name → pointer (for execute_v2)
#             # We no longer call binding.data_ptr(); we stored 'ptr' in our namedtuple
#             self.bindings_addr = collections.OrderedDict(
#                 (name, binding.ptr) for name, binding in self.bindings.items()
#             )

#             # Store input and output binding names
#             self.input_names = self._get_input_names()
#             self.output_names = self._get_output_names()

#             if self.backend != "torch":
#                 # We only support "torch" backend here
#                 raise ValueError(f"Unsupported backend: {self.backend}")

#             # A small time profiler (optional)
#             self.time_profile = TimeProfiler()

#         except Exception as e:
#             # On error, attempt to free any partial resources
#             self.destroy()
#             if self.logger:
#                 self.logger.error(f"Error during TRTInference initialization: {e}")
#             raise

#     def _get_input_names(self):
#         """
#         Return a list of binding names for I/O tensors that TensorRT sees as 'INPUT'.
#         """
#         names = []
#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             mode = self.engine.get_tensor_mode(name)
#             if mode == trt.TensorIOMode.INPUT:
#                 names.append(name)
#         return names

#     def _get_output_names(self):
#         """
#         Return a list of binding names for I/O tensors that TensorRT sees as 'OUTPUT'.
#         """
#         names = []
#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             mode = self.engine.get_tensor_mode(name)
#             if mode == trt.TensorIOMode.OUTPUT:
#                 names.append(name)
#         return names

#     def _allocate_bindings(self, engine, context, max_batch_size, device):
#         """
#         Allocate a torch.Tensor for each binding in the engine.
#         If a dimension is -1, we replace dim[0] with max_batch_size and dim[other] with 1.
#         Returns OrderedDict[name → Binding(namedtuple)].
#         """
#         Binding = collections.namedtuple(
#             "Binding", ("name", "dtype", "shape", "data", "ptr")
#         )
#         bindings = collections.OrderedDict()

#         for i in range(engine.num_io_tensors):
#             name = engine.get_tensor_name(i)
#             shape = list(engine.get_tensor_shape(name))
#             trt_dtype = engine.get_tensor_dtype(name)

#             # Map TRT dtype → numpy dtype → torch dtype
#             if trt_dtype == trt.DataType.BOOL:
#                 np_dtype = np.bool_
#             else:
#                 np_dtype = trt.nptype(trt_dtype)

#             is_input = (engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)

#             # Replace any -1 dims
#             for dim_idx, dim_val in enumerate(shape):
#                 if dim_val == -1:
#                     if dim_idx == 0:
#                         shape[dim_idx] = max_batch_size
#                         if is_input:
#                             self.dynamic_input_shapes[name] = True
#                     else:
#                         shape[dim_idx] = 1

#             # Create a torch tensor on 'device'
#             torch_dtype = torch.bool if np_dtype == np.bool_ else torch.from_numpy(
#                 np.empty((), dtype=np_dtype)).dtype
#             tensor = torch.from_numpy(np.empty(tuple(shape), dtype=np_dtype)).to(device)

#             bindings[name] = Binding(
#                 name=name,
#                 dtype=np_dtype,
#                 shape=tuple(shape),
#                 data=tensor,
#                 ptr=int(tensor.data_ptr())
#             )

#         return bindings

#     def run_torch(self, blob: dict):
#         """
#         Single-image inference (batch_size=1) in 'torch' backend:
#           - Copy input tensor(s) into the pre-allocated binding
#           - Execute context.execute_v2()
#           - Slice outputs back to proper shapes → return outputs dict
#         """
#         current_batch_size = 1

#         # 1) Copy each input tensor into the binding
#         for name in self.input_names:
#             if name not in blob:
#                 raise ValueError(f"Input blob missing key: {name}. Keys: {blob.keys()}")
#             input_tensor = blob[name]
#             if input_tensor.shape[0] != current_batch_size:
#                 raise ValueError(f"Expected batch_size=1, got {input_tensor.shape[0]}")

#             binding_idx = self.engine.get_binding_index(name)
#             # If dynamic, set binding shape
#             if name in self.dynamic_input_shapes:
#                 desired_shape = tuple(input_tensor.shape)
#                 if tuple(self.context.get_binding_shape(binding_idx)) != desired_shape:
#                     if not self.context.set_binding_shape(binding_idx, desired_shape):
#                         raise RuntimeError(f"Failed to set dynamic shape for {name} to {desired_shape}")
#                     # Update our stored binding shape
#                     old_binding = self.bindings[name]
#                     self.bindings[name] = old_binding._replace(shape=desired_shape)

#             # Copy data into the preallocated tensor
#             target_dtype = torch.bool if self.bindings[name].dtype == np.bool_ else self.bindings[name].data.dtype
#             if input_tensor.dtype != target_dtype:
#                 input_tensor = input_tensor.to(target_dtype)

#             # Use the first slice (since batch_size=1)
#             self.bindings[name].data[0:current_batch_size].copy_(input_tensor)
#             self.bindings_addr[name] = int(self.bindings[name].data[0:current_batch_size].data_ptr())

#         # 2) Execute
#         self.context.execute_v2(list(self.bindings_addr.values()))

#         # 3) Gather outputs into a dict
#         outputs = {}
#         for name in self.output_names:
#             binding_idx = self.engine.get_binding_index(name)
#             out_shape = tuple(self.context.get_binding_shape(binding_idx))
#             tensor_full = self.bindings[name].data
#             # Slice off the "max_batch_size" dimension to [0:1, ...]
#             slices = [slice(0, dim) for dim in out_shape]
#             outputs[name] = tensor_full[tuple(slices)].clone()

#         return outputs

#     def __call__(self, blob: dict):
#         if self.backend == "torch":
#             return self.run_torch(blob)
#         else:
#             raise ValueError(f"Unsupported backend: {self.backend}")

#     def destroy(self):
#         """
#         Clean up all TRT-related contexts and engine instances.
#         """
#         if self.logger:
#             self.logger.info("Destroying TRTInference resources and freeing GPU memory...")

#         # Delete bindings (which hold torch tensors). Let GC free them.
#         if hasattr(self, "bindings"):
#             try:
#                 del self.bindings
#             except Exception:
#                 pass
#             self.bindings = None

#         if hasattr(self, "bindings_addr"):
#             try:
#                 del self.bindings_addr
#             except Exception:
#                 pass
#             self.bindings_addr = None

#         # Delete context, engine, runtime
#         if hasattr(self, "context") and self.context is not None:
#             try:
#                 del self.context
#             except Exception:
#                 pass
#             self.context = None

#         if hasattr(self, "engine") and self.engine is not None:
#             try:
#                 del self.engine
#             except Exception:
#                 pass
#             self.engine = None

#         if hasattr(self, "runtime") and self.runtime is not None:
#             try:
#                 del self.runtime
#             except Exception:
#                 pass
#             self.runtime = None

#         # Clear CUDA cache if on GPU
#         try:
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 if self.logger:
#                     self.logger.info("CUDA cache cleared.")
#         except Exception:
#             if self.logger:
#                 self.logger.warning("Exception while clearing CUDA cache during destroy().")

#     def __del__(self):
#         # Ensure resources are freed if the object is garbage-collected
#         try:
#             self.destroy()
#         except Exception:
#             pass



# trt_inference.py

import collections
import torch
import numpy as np
import tensorrt as trt
import contextlib
import time


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference:
    """
    Wrapper around a TensorRT engine that:
      - Deserializes an engine from a file
      - Allocates input/output bindings as Torch tensors
      - Executes inference (single batch of size 1)
      - Returns a dict of output tensors
    
    Updated for TensorRT 8.6+ compatibility (removes deprecated get_binding_index)
    """

    def __init__(self,
                 engine_path: str,
                 device: str = "cuda:0",
                 backend: str = "torch",
                 max_batch_size: int = 32,
                 verbose: bool = False,
                 logger = None):
        """
        :param engine_path: Path to the serialized TensorRT .engine file
        :param device: CUDA device string ("cuda:0") or "cpu"
        :param backend: Only "torch" is supported in this implementation
        :param max_batch_size: Maximum batch size to allocate for dynamic dimensions
        :param verbose: If True, create a verbose TRT logger
        :param logger: Optional Python logger to record messages
        """
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.backend = backend
        self.max_batch_size_for_bindings = max_batch_size
        self.logger = logger

        # Prepare a TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        # TRT runtime/engine/context placeholders
        self.runtime = None
        self.engine = None
        self.context = None

        # These will be filled by get_bindings()
        self.dynamic_input_shapes = {}
        self.bindings = collections.OrderedDict()
        self.bindings_addr = collections.OrderedDict()
        self.input_names = []
        self.output_names = []

        # Store tensor name to index mapping for newer TensorRT versions
        self.tensor_name_to_idx = {}

        # Attempt to build/deserializes the engine and create context
        try:
            if self.logger:
                self.logger.info(f"Initializing TensorRT runtime and deserializing engine from: {self.engine_path}")
            trt.init_libnvinfer_plugins(self.trt_logger, "")

            # Create TRT runtime
            self.runtime = trt.Runtime(self.trt_logger)

            # Deserialize engine
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")

            # Create execution context
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            # Build tensor name mapping for newer TensorRT versions
            self._build_tensor_mappings()

            # Build binding data structures (torch tensors) for each I/O
            self.bindings = self._allocate_bindings(self.engine,
                                                     self.context,
                                                     self.max_batch_size_for_bindings,
                                                     self.device)

            # Build a map from binding name → pointer (for execute_v2)
            # We no longer call binding.data_ptr(); we stored 'ptr' in our namedtuple
            self.bindings_addr = collections.OrderedDict(
                (name, binding.ptr) for name, binding in self.bindings.items()
            )

            # Store input and output binding names
            self.input_names = self._get_input_names()
            self.output_names = self._get_output_names()

            if self.backend != "torch":
                # We only support "torch" backend here
                raise ValueError(f"Unsupported backend: {self.backend}")

            # A small time profiler (optional)
            self.time_profile = TimeProfiler()

        except Exception as e:
            # On error, attempt to free any partial resources
            self.destroy()
            if self.logger:
                self.logger.error(f"Error during TRTInference initialization: {e}")
            raise

    def _build_tensor_mappings(self):
        """Build mapping from tensor names to indices for TensorRT compatibility"""
        self.tensor_name_to_idx = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_name_to_idx[name] = i

    def _get_binding_index_safe(self, name):
        """
        Safe method to get binding index that works with both old and new TensorRT versions
        """
        # Try the modern approach first
        if hasattr(self.engine, 'get_tensor_name') and name in self.tensor_name_to_idx:
            return self.tensor_name_to_idx[name]
        
        # Fallback to deprecated method if available
        if hasattr(self.engine, 'get_binding_index'):
            try:
                return self.engine.get_binding_index(name)
            except:
                pass
        
        # If both fail, raise an error
        raise RuntimeError(f"Could not find binding index for tensor: {name}")

    def _get_binding_shape_safe(self, name_or_idx):
        """
        Safe method to get binding shape that works with both old and new TensorRT versions
        """
        if isinstance(name_or_idx, str):
            name = name_or_idx
            # Try modern tensor-based approach
            if hasattr(self.context, 'get_tensor_shape'):
                try:
                    return tuple(self.context.get_tensor_shape(name))
                except:
                    pass
            
            # Fallback to index-based approach
            try:
                idx = self._get_binding_index_safe(name)
                if hasattr(self.context, 'get_binding_shape'):
                    return tuple(self.context.get_binding_shape(idx))
            except:
                pass
        else:
            idx = name_or_idx
            if hasattr(self.context, 'get_binding_shape'):
                try:
                    return tuple(self.context.get_binding_shape(idx))
                except:
                    pass
        
        raise RuntimeError(f"Could not get binding shape for: {name_or_idx}")

    def _set_binding_shape_safe(self, name_or_idx, shape):
        """
        Safe method to set binding shape that works with both old and new TensorRT versions
        """
        if isinstance(name_or_idx, str):
            name = name_or_idx
            # Try modern tensor-based approach
            if hasattr(self.context, 'set_input_shape'):
                try:
                    return self.context.set_input_shape(name, shape)
                except:
                    pass
            
            # Fallback to index-based approach
            try:
                idx = self._get_binding_index_safe(name)
                if hasattr(self.context, 'set_binding_shape'):
                    return self.context.set_binding_shape(idx, shape)
            except:
                pass
        else:
            idx = name_or_idx
            if hasattr(self.context, 'set_binding_shape'):
                try:
                    return self.context.set_binding_shape(idx, shape)
                except:
                    pass
        
        return False

    def _get_input_names(self):
        """
        Return a list of binding names for I/O tensors that TensorRT sees as 'INPUT'.
        """
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def _get_output_names(self):
        """
        Return a list of binding names for I/O tensors that TensorRT sees as 'OUTPUT'.
        """
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def _allocate_bindings(self, engine, context, max_batch_size, device):
        """
        Allocate a torch.Tensor for each binding in the engine.
        If a dimension is -1, we replace dim[0] with max_batch_size and dim[other] with 1.
        Returns OrderedDict[name → Binding(namedtuple)].
        """
        Binding = collections.namedtuple(
            "Binding", ("name", "dtype", "shape", "data", "ptr")
        )
        bindings = collections.OrderedDict()

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = list(engine.get_tensor_shape(name))
            trt_dtype = engine.get_tensor_dtype(name)

            # Map TRT dtype → numpy dtype → torch dtype
            if trt_dtype == trt.DataType.BOOL:
                np_dtype = np.bool_
            else:
                np_dtype = trt.nptype(trt_dtype)

            is_input = (engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)

            # Replace any -1 dims
            for dim_idx, dim_val in enumerate(shape):
                if dim_val == -1:
                    if dim_idx == 0:
                        shape[dim_idx] = max_batch_size
                        if is_input:
                            self.dynamic_input_shapes[name] = True
                    else:
                        shape[dim_idx] = 1

            # Create a torch tensor on 'device'
            torch_dtype = torch.bool if np_dtype == np.bool_ else torch.from_numpy(
                np.empty((), dtype=np_dtype)).dtype
            tensor = torch.from_numpy(np.empty(tuple(shape), dtype=np_dtype)).to(device)

            bindings[name] = Binding(
                name=name,
                dtype=np_dtype,
                shape=tuple(shape),
                data=tensor,
                ptr=int(tensor.data_ptr())
            )

        return bindings

    def run_torch(self, blob: dict):
        """
        Single-image inference (batch_size=1) in 'torch' backend:
          - Copy input tensor(s) into the pre-allocated binding
          - Execute context.execute_v2()
          - Slice outputs back to proper shapes → return outputs dict
        """
        current_batch_size = 1

        # 1) Copy each input tensor into the binding
        for name in self.input_names:
            if name not in blob:
                raise ValueError(f"Input blob missing key: {name}. Keys: {blob.keys()}")
            input_tensor = blob[name]
            if input_tensor.shape[0] != current_batch_size:
                raise ValueError(f"Expected batch_size=1, got {input_tensor.shape[0]}")

            # If dynamic, set binding shape using safe method
            if name in self.dynamic_input_shapes:
                desired_shape = tuple(input_tensor.shape)
                current_shape = self._get_binding_shape_safe(name)
                
                if current_shape != desired_shape:
                    if not self._set_binding_shape_safe(name, desired_shape):
                        raise RuntimeError(f"Failed to set dynamic shape for {name} to {desired_shape}")
                    # Update our stored binding shape
                    old_binding = self.bindings[name]
                    self.bindings[name] = old_binding._replace(shape=desired_shape)

            # Copy data into the preallocated tensor
            target_dtype = torch.bool if self.bindings[name].dtype == np.bool_ else self.bindings[name].data.dtype
            if input_tensor.dtype != target_dtype:
                input_tensor = input_tensor.to(target_dtype)

            # Use the first slice (since batch_size=1)
            self.bindings[name].data[0:current_batch_size].copy_(input_tensor)
            self.bindings_addr[name] = int(self.bindings[name].data[0:current_batch_size].data_ptr())

        # 2) Execute using modern tensor-based API
        if hasattr(self.context, 'execute_v3'):
            # TensorRT 10+ with execute_v3
            success = self.context.execute_v3(1)  # batch_size = 1
        else:
            # TensorRT 8.x/9.x with execute_v2
            success = self.context.execute_v2(list(self.bindings_addr.values()))
            
        if not success:
            raise RuntimeError("TensorRT execution failed")

        # 3) Gather outputs into a dict
        outputs = {}
        for name in self.output_names:
            out_shape = self._get_binding_shape_safe(name)
            tensor_full = self.bindings[name].data
            # Slice off the "max_batch_size" dimension to [0:1, ...]
            slices = [slice(0, dim) for dim in out_shape]
            outputs[name] = tensor_full[tuple(slices)].clone()

        return outputs

    def __call__(self, blob: dict):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def destroy(self):
        """
        Clean up all TRT-related contexts and engine instances.
        """
        if self.logger:
            self.logger.info("Destroying TRTInference resources and freeing GPU memory...")

        # Delete bindings (which hold torch tensors). Let GC free them.
        if hasattr(self, "bindings"):
            try:
                del self.bindings
            except Exception:
                pass
            self.bindings = None

        if hasattr(self, "bindings_addr"):
            try:
                del self.bindings_addr
            except Exception:
                pass
            self.bindings_addr = None

        # Delete context, engine, runtime
        if hasattr(self, "context") and self.context is not None:
            try:
                del self.context
            except Exception:
                pass
            self.context = None

        if hasattr(self, "engine") and self.engine is not None:
            try:
                del self.engine
            except Exception:
                pass
            self.engine = None

        if hasattr(self, "runtime") and self.runtime is not None:
            try:
                del self.runtime
            except Exception:
                pass
            self.runtime = None

        # Clear CUDA cache if on GPU
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.logger:
                    self.logger.info("CUDA cache cleared.")
        except Exception:
            if self.logger:
                self.logger.warning("Exception while clearing CUDA cache during destroy().")

    def __del__(self):
        # Ensure resources are freed if the object is garbage-collected
        try:
            self.destroy()
        except Exception:
            pass