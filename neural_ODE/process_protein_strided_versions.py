  #THIS VERSION DOES THE GRADIENT STEP EVERY CHUNK, SO IT HAS HIGHER MEMORY SINCE THE COMPUTATION GRAPH IS NOT FREED UNTIL THE END

    def _process_protein_strided(self, protein_id: str, data_dir: str, model: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer, scaler: GradScaler, training: bool) -> Dict:
        """Sequential strided processing with per-chunk backward passes"""
        m_init, z_init, selected_blocks = DataManager.load_protein_blocks(
            protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
            strided=True, block_stride=self.config.prelim_block_stride, max_blocks=49
        )

        chunk_size = self.config.prelim_chunk_size
        total_loss = 0.0  # Scalar accumulator
        total_chunks = 0

        # Start with initial state
        current_m, current_z = m_init, z_init
        current_block = selected_blocks[0]

        # Process sequential chunks with individual backward passes
        for i in range(0, len(selected_blocks) - 1, chunk_size):
            # Get target blocks for this chunk
            target_start = i + 1
            target_end = min(target_start + chunk_size, len(selected_blocks))
            target_blocks = selected_blocks[target_start:target_end]

            if not target_blocks:
                break

            # Reset gradients for this chunk
            if training:
                optimizer.zero_grad()

            # Create time points from current block to targets
            time_points = [float(current_block) / 48.0]
            for block in target_blocks:
                time_points.append(float(block) / 48.0)

            chunk_time_points = torch.tensor(time_points, device=self.config.device)
            print("chunk_time_points", chunk_time_points)
            print("target_blocks", target_blocks)
            print("i",i)
            with autocast(enabled=self.config.use_amp):
                trajectory = odeint(
                    model, (current_m, current_z), chunk_time_points,
                    method=self.config.integrator, rtol=1e-4, atol=1e-5
                )

                # Supervise each prediction
                chunk_loss = 0
                chunk_steps = 0

                for j, target_block in enumerate(target_blocks):
                    print("j",j)
                    print("target_block", target_block)
                    pred_m = trajectory[0][j + 1]  # +1 to skip starting state
                    pred_z = trajectory[1][j + 1]

                    # Load target
                    m_target, z_target = DataManager.load_protein_blocks(
                        protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
                        target_block=target_block
                    )

                    step_loss = self.compute_adaptive_loss(pred_m, m_target, pred_z, z_target)
                    chunk_loss += step_loss
                    chunk_steps += 1

                    # Save last target as next starting state (detached to break graph)
                    if j == len(target_blocks) - 1:  # Last target in this chunk
                        current_m = m_target.detach().clone()
                        current_z = z_target.detach().clone()
                        current_block = target_block

                    del m_target, z_target

                # Average chunk loss
                if chunk_steps > 0:
                    chunk_loss = chunk_loss / chunk_steps

                    # Backward pass for this chunk only
                    if training:
                        if self.config.use_amp:
                            scaler.scale(chunk_loss).backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            chunk_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    # Store scalar loss for reporting
                    chunk_loss_scalar = chunk_loss.item()
                    print(
                        f"  📦 Chunk {total_chunks + 1} (from block {selected_blocks[i]} → predict {target_blocks}): loss = {chunk_loss_scalar:.6f}")

                    total_loss += chunk_loss_scalar
                    total_chunks += 1

            # Aggressive cleanup after each chunk
            del trajectory, chunk_time_points, chunk_loss
            torch.cuda.empty_cache()

        if total_chunks == 0:
            return {'protein': protein_id, 'loss': 0.0, 'error': 'no_valid_chunks'}

        avg_loss = total_loss / total_chunks

        # Final cleanup
        del m_init, z_init, current_m, current_z

        return {
            'protein': protein_id, 'loss': avg_loss, 'selected_blocks': selected_blocks,
            'num_chunks': total_chunks, 'cluster_size': self.config.reduced_cluster_size,
            'approach': 'per_chunk_backward_sequential_strided'
        }

#   THIS VERSION DOES THE GRADIENT STEP EVERY CHUNK, SO IT HAS HIGHER MEMORY SINCE THE COMPUTATION GRAPH IS NOT FREED UNTIL THE END
  def _process_protein_strided(self, protein_id: str, data_dir: str, model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer, scaler: GradScaler, training: bool) -> Dict:
      """Sequential strided processing with batch gradient updates and memory safety"""
      m_init, z_init, selected_blocks = DataManager.load_protein_blocks(
          protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
          strided=True, block_stride=self.config.prelim_block_stride, max_blocks=49
      )

      if training:
          optimizer.zero_grad()  # ← ONCE at start (like original)

      chunk_size = self.config.prelim_chunk_size
      accumulated_loss = None  # ← Accumulate actual loss tensors for gradients
      total_loss_scalar = 0.0  # ← Separate scalar accumulator for reporting
      total_chunks = 0

      # Start with initial state
      current_m, current_z = m_init, z_init
      current_block = selected_blocks[0]

      # Process sequential chunks (accumulate gradients, don't step)
      for i in range(0, len(selected_blocks) - 1, chunk_size):
          # Get target blocks for this chunk
          target_start = i + 1
          target_end = min(target_start + chunk_size, len(selected_blocks))
          target_blocks = selected_blocks[target_start:target_end]

          if not target_blocks:
              break

          # Create time points from current block to targets
          time_points = [float(current_block) / 48.0]
          for block in target_blocks:
              time_points.append(float(block) / 48.0)

          chunk_time_points = torch.tensor(time_points, device=self.config.device)

          with autocast(enabled=self.config.use_amp):
              trajectory = odeint(
                  model, (current_m, current_z), chunk_time_points,
                  method=self.config.integrator, rtol=1e-4, atol=1e-5
              )

              # Supervise each prediction
              chunk_loss = 0
              chunk_steps = 0

              for j, target_block in enumerate(target_blocks):
                  pred_m = trajectory[0][j + 1]  # +1 to skip starting state
                  pred_z = trajectory[1][j + 1]

                  # Load target
                  m_target, z_target = DataManager.load_protein_blocks(
                      protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
                      target_block=target_block
                  )

                  step_loss = self.compute_adaptive_loss(pred_m, m_target, pred_z, z_target)
                  chunk_loss += step_loss  # ← Keep as tensor for gradients
                  chunk_steps += 1

                  # Save last target as next starting state (detached to save memory)
                  if j == len(target_blocks) - 1:  # Last target in this chunk
                      current_m = m_target.detach().clone()
                      current_z = z_target.detach().clone()
                      current_block = target_block

                  del m_target, z_target, step_loss  # ← Cleanup intermediate tensors

              # Average chunk loss
              if chunk_steps > 0:
                  chunk_loss = chunk_loss / chunk_steps

                  # Accumulate loss tensor for batch gradient (like original)
                  if accumulated_loss is None:
                      accumulated_loss = chunk_loss
                  else:
                      accumulated_loss = accumulated_loss + chunk_loss  # ← Tensor accumulation

                  # Store scalar for reporting
                  chunk_loss_scalar = chunk_loss.item()
                  print(
                      f"  📦 Chunk {total_chunks + 1} (from block {selected_blocks[i]} → predict {target_blocks}): loss = {chunk_loss_scalar:.6f}")

                  total_loss_scalar += chunk_loss_scalar
                  total_chunks += 1

          # Memory cleanup: delete chunk tensors but keep accumulated_loss
          del trajectory, chunk_time_points, chunk_loss
          torch.cuda.empty_cache()

      # Single batch backward pass at the end (like original)
      if total_chunks > 0:
          avg_loss = accumulated_loss / total_chunks  # ← Still a tensor with gradients

          if training:
              if self.config.use_amp:
                  scaler.scale(avg_loss).backward()  # ← ONE backward pass
                  scaler.unscale_(optimizer)
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                  scaler.step(optimizer)  # ← ONE optimizer step
                  scaler.update()
              else:
                  avg_loss.backward()  # ← ONE backward pass
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                  optimizer.step()  # ← ONE optimizer step

              loss_value = avg_loss.item()
          else:
              loss_value = avg_loss.item()

          del accumulated_loss, avg_loss
      else:
          loss_value = 0.0

      # Final cleanup
      del m_init, z_init, current_m, current_z

      return {
          'protein': protein_id, 'loss': loss_value, 'selected_blocks': selected_blocks,
          'num_chunks': total_chunks, 'cluster_size': self.config.reduced_cluster_size,
          'approach': 'batch_update_sequential_strided'
      }